# Copyright 2023 The Self-Align Team
# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import gc
import logging
import os
import pathlib
from pathlib import Path
import re
import math
from collections import OrderedDict
from functools import partial
from itertools import chain
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tqdm

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
from torch.distributed.fsdp import MixedPrecision, CPUOffload, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from grading import grader

try:
    import wandb
except ImportError:
    wandb = None

import data_utils.common_utils as common_utils
from data_utils.data_utils_ppo import QueryDataset
from data_utils.tokenizer_utils import FakePreTrainedTokenizer

import models.rl_model as rl_model
from models.model import TransformerBlock, set_global_compile_mode
from models.reward_model import RewardModel, apply_reward_modeling_head
from models.tp import (
    apply_reward_head_tp,
    get_data_parallel_rank,
    get_data_parallel_group,
    get_data_parallel_world_size,
    get_model_parallel_rank,
    get_model_parallel_world_size,
)
from models.rl_trainer import RLTrainer, truncate_after_eos, truncate_after_eos_padded

from training_utils.training_args import TrainingArguments
from training_utils.checkpoint_hook import (
    checkpoint_hook,
    get_latest_checkpoint_path,
    load_checkpoint,
    load_model_from_from_ckpt,
    load_reward_model_from_ckpt,
)

AnyPath = Union[str, os.PathLike, pathlib.Path]
AnyPathOrNone = Optional[AnyPath]

logger = logging.getLogger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
POLICY_MODEL_NAME = "policy_model.pt"
POLICY_OPTIMIZER_NAME = "policy_optimizer.pt"
POLICY_SCHEDULER_NAME = "policy_scheduler.pt"
VALUE_MODEL_NAME = "value_model.pt"
VALUE_OPTIMIZER_NAME = "value_optimizer.pt"
VALUE_SCHEDULER_NAME = "value_scheduler.pt"
SCALER_NAME = "scaler.pt"


class PPOTrainer(RLTrainer):
    def __init__(
        self,
        args: TrainingArguments,
        train_dataset: QueryDataset,
        data_collator: Callable,
        policy: rl_model.Policy,
        value_model: rl_model.Value,
        ref_policy: rl_model.Policy,
        reward_model: RewardModel,
        tokenizer: FakePreTrainedTokenizer,
        eval_dataset: Optional[QueryDataset] = None,
        test_dataset: Optional[QueryDataset] = None,
        policy_optimizer: Optional[torch.optim.Optimizer] = None,
        policy_lr_scheduler: Optional[LRScheduler] = None,
        value_optimizer: Optional[torch.optim.Optimizer] = None,
        value_lr_scheduler: Optional[LRScheduler] = None,
        reward_tokenizer: Optional[FakePreTrainedTokenizer] = None,
        unwrapped_policy: Optional[rl_model.Policy] = None,
    ):
        if reward_tokenizer is not None:
            raise NotImplementedError("Seperate reward tokenizer is not supported yet.")

        super(PPOTrainer, self).__init__(
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
            data_collator=data_collator,
            policy=policy,
            value_model=value_model,
            ref_policy=ref_policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            policy_optimizer=policy_optimizer,
            policy_lr_scheduler=policy_lr_scheduler,
            value_optimizer=value_optimizer,
            value_lr_scheduler=value_lr_scheduler,
            reward_tokenizer=reward_tokenizer,
            unwrapped_policy=unwrapped_policy,
        )

    def _shape_reward(
        self,
        rewards: torch.Tensor,
        responses: torch.Tensor,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        reg_entropies: torch.Tensor,
        terminal_reward: bool = True,
    ) -> Dict[str, torch.Tensor]:
        # For some reason, line below doesn't work.
        # kl = (logits.softmax(dim=-1) * (logits.log_softmax(dim=-1) - ref_logits.log_softmax(dim=-1))).sum(dim=-1)

        if self.args.kl_approximator == "k1":
            # KL (q | p) = sum_i q_i (log q_i - log p_i)
            kl = torch.clamp(logprobs - ref_logprobs, min=0.0)
        elif self.args.kl_approximator == "k2":
            kl = logprobs - ref_logprobs
        elif self.args.kl_approximator == "k3":
            # r = p / q, log r = log p - log q
            # KL (q | p) = (r - 1) - log r = e ^ log r - 1 - log r
            log_r = ref_logprobs - logprobs
            kl = torch.exp(log_r) - 1.0 - log_r
        elif self.args.kl_approximator == "maxent":
            kl = logprobs
        else:
            raise ValueError(f"Unknown KL approximator: {self.args.kl_approximator}")

        non_score_rewards = -self.kl_ctl.value * kl

        if self.args.maxent_coef > 0:
            non_score_rewards = (
                non_score_rewards + self.args.maxent_coef * reg_entropies
            )

        shaped_rewards = non_score_rewards.clone()
        # This introduces a small index off by one bug if pad_token_id == eos_token_id.
        terminal_positions = (responses != self.tokenizer.pad_id).sum(dim=1) - 1

        if self.args.min_seq_len > 0 and self.args.min_seq_len_coef > 0:
            # min_seq_penalty = max(min_seq_len - seq_len, 0) * min_seq_len_coef
            min_seq_penalty = (
                torch.clamp(self.args.min_seq_len - terminal_positions, min=0.0).float()
                * self.args.min_seq_len_coef
            )
            rewards = rewards - min_seq_penalty

        if terminal_reward:
            shaped_rewards[
                torch.arange(rewards.size(0), device=responses.device),
                terminal_positions,
            ] += rewards
        else:
            shaped_rewards = shaped_rewards + rewards
        return dict(
            shaped_rewards=shaped_rewards, non_score_rewards=non_score_rewards, kl=kl
        )

    def _estimate_advantage(
        self, rewards: torch.Tensor, values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Generalized advantage estimation.

        Reference:
            https://arxiv.org/abs/1506.02438
        """
        if self.args.whiten_rewards:
            rewards = whiten(
                rewards, shift_mean=False, async_stats=self.args.whitening_async_stats
            )
        else:
            rewards = rewards * 50.0
        lastgaelam = 0
        advantages_reversed = []
        gen_length = self.args.target_max_len
        for t in reversed(range(gen_length)):
            nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
            delta = rewards[:, t] + self.args.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = whiten(
            advantages, shift_mean=True, async_stats=self.args.whitening_async_stats
        )
        return dict(returns=returns, advantages=advantages)

    def _calculate_outcome_accuracy(
        self,
        predicted_answers: List[str],
        gt_answers: List[str],
        answers: List[str],
        levels: List[int],
    ):
        assert len(predicted_answers) == len(answers)

        assert not (
            self.args.outcome_reward and self.args.easy_outcome_reward
        ), "Cannot use both outcome_reward and easy_outcome_reward."

        with common_utils.DisableLogger():
            outcome_accuracy = [
                1.0 if grader.grade_answer(predicted_answer, gt_answer) else 0.0
                for predicted_answer, gt_answer in zip(predicted_answers, gt_answers)
            ]

            # TODO (zhiqings): 0.25 is a magic number.
            unavailable_reward = 0.25
            if self.args.outcome_reward:
                symbolic_rewards = outcome_accuracy
            elif self.args.easy_outcome_reward:
                symbolic_rewards = []
                for predicted_answer, answer in zip(predicted_answers, answers):
                    if answer == "Unavailable":
                        score = unavailable_reward
                    elif grader.grade_answer(predicted_answer, answer):
                        score = 1.0
                    else:
                        score = 0.0
                    symbolic_rewards.append(score)
            else:
                symbolic_rewards = [
                    unavailable_reward for _ in range(len(predicted_answers))
                ]

        assert len(symbolic_rewards) == len(predicted_answers)

        per_level_counts = {}
        per_level_accuracy = {}

        all_unique_levels = list(range(1, 6))

        for level in all_unique_levels:
            per_level_counts[level] = []
            per_level_accuracy[level] = []

        for level, accuracy in zip(levels, outcome_accuracy):
            for unique_level in all_unique_levels:
                if level == unique_level:
                    per_level_counts[unique_level].append(1.0)
                    per_level_accuracy[unique_level].append(accuracy)
                else:
                    per_level_counts[unique_level].append(0.0)
                    per_level_accuracy[unique_level].append(0.0)

        for level in all_unique_levels:
            assert len(per_level_counts[level]) == len(outcome_accuracy)
            assert len(per_level_accuracy[level]) == len(outcome_accuracy)
            per_level_counts[level] = torch.tensor(
                per_level_counts[level], device=self.device
            )
            per_level_accuracy[level] = torch.tensor(
                per_level_accuracy[level], device=self.device
            )

        original_symbolic_rewards = symbolic_rewards

        symbolic_rewards = torch.tensor(symbolic_rewards, device=self.device)
        outcome_accuracy = torch.tensor(outcome_accuracy, device=self.device)

        ret_dict = {
            "symbolic_rewards": symbolic_rewards,
            "outcome_accuracy": outcome_accuracy,
        }

        for level in sorted(list(all_unique_levels)):
            ret_dict[f"level_{level}_counts"] = per_level_counts[level]
            ret_dict[f"level_{level}_accuracy"] = per_level_accuracy[level]

        return ret_dict, original_symbolic_rewards

    def rollout(self, queries_data) -> Dict[str, torch.Tensor]:
        """Rollout trajectories with policy.

        Args:
            queries_data: Sequence of batches or DataLoader.
                Each batch is a dict with keys 'queries', 'query_attn_masks', and 'answers'.

        Returns:
            Dictionary with keys
                'queries', 'query_attn_masks', 'responses',
                'logprobs', 'ref_logprobs', 'values',
                'rewards', 'non_score_rewards', 'shaped_rewards'.
        """
        # Give up dropout throughout.
        self.policy: rl_model.Policy = self.policy.eval()
        self.value_model: rl_model.Value = self.value_model.eval()
        self.ref_policy: rl_model.Policy = self.ref_policy.eval()
        self.reward_model: RewardModel = self.reward_model.eval()

        gc.collect()
        torch.cuda.empty_cache()

        unwrapped_policy: rl_model.Policy = self.policy
        if self.unwrapped_policy is not None:
            for _ in tqdm.tqdm(
                range(1), disable=not self.is_main_process, desc="unwrapped_policy"
            ):
                unwrapped_policy = self.unwrapped_policy

                cfg = FullStateDictConfig(
                    offload_to_cpu=self.args.fsdp_consolidate_cpu_offload,
                    rank0_only=False,
                )
                optim_cfg = FullOptimStateDictConfig(
                    offload_to_cpu=self.args.fsdp_consolidate_cpu_offload,
                    rank0_only=False,
                )

                with FSDP.state_dict_type(
                    module=self.policy.base_model,
                    state_dict_type=StateDictType.FULL_STATE_DICT,
                    state_dict_config=cfg,
                    optim_state_dict_config=optim_cfg,
                ):
                    policy_model_state_dict = self.policy.base_model.state_dict()

                if list(policy_model_state_dict.keys())[0].startswith("module."):
                    assert all(
                        [
                            k.startswith("module.")
                            for k in policy_model_state_dict.keys()
                        ]
                    )
                    policy_model_state_dict = {
                        k[len("module.") :]: v
                        for k, v in policy_model_state_dict.items()
                    }

                policy_model_state_dict = {
                    k: v.to(
                        dtype=self.args.compute_dtype,
                        device=self.device,
                        non_blocking=self.args.fsdp_consolidate_cpu_offload,
                    )
                    for k, v in policy_model_state_dict.items()
                    if "attention.kv_cache" not in k
                }

                unwrapped_policy.base_model.load_state_dict(
                    policy_model_state_dict,
                    strict=False,  # Allow missing keys for attention.kv_cache
                )
                del policy_model_state_dict
                unwrapped_policy: rl_model.Policy = unwrapped_policy.eval()

        if self.args.compile:
            all_backward_hooks = remove_all_backward_hooks(unwrapped_policy)
            setting_flag = False
            for _, module in unwrapped_policy.named_modules():
                if hasattr(module, "in_compile_mode"):
                    if not module.in_compile_mode:
                        setting_flag = True
                        module.in_compile_mode = True
            if setting_flag:
                rank0_print("Setting compile mode of unwrapped_policy to True")

        rollouts: List[Dict[str, torch.Tensor]] = []
        with torch.no_grad():
            for batch_idx, batch in tqdm.tqdm(
                enumerate(queries_data),
                total=len(queries_data),
                disable=not self.is_main_process,
                desc="rollout",
            ):
                # Sample rollouts.
                (
                    queries,
                    query_attn_masks,
                    answer_gt_levels,
                ) = common_utils.unpack_dict(
                    common_utils.prepare_inputs(batch, device=self.device),
                    keys=(
                        "queries",
                        "query_attn_masks",
                        "answers",
                    ),
                )

                queries = self.prepare_tp_batch(queries)
                query_attn_masks = self.prepare_tp_batch(query_attn_masks)
                answer_gt_levels = self.prepare_tp_batch(answer_gt_levels)

                with torch.cuda.amp.autocast(dtype=self.args.compute_dtype):
                    responses = unwrapped_policy.respond(
                        queries, query_attn_masks, temperature=self.args.temperature
                    )["responses"]

                # Choose token count per batch such that tile count is multiple of SM count
                # The NVIDIA A100 GPU contains 108 SMs
                # The H100 SXM5 GPU has 132 SMs, and the PCIe version has 114 SMs.
                assert queries.size(1) % 8 == 0, (
                    "Please enable Tensor Cores with batch size a multiple of 8, "
                    "not {}.".format(queries.size(1))
                )
                assert responses.size(1) % 8 == 0, (
                    "Please enable Tensor Cores with batch size a multiple of 8, "
                    "not {}.".format(responses.size(1))
                )

                rollouts_batch = {
                    "queries": queries,
                    "query_attn_masks": query_attn_masks,
                    "responses": responses,
                }

                text_queries = self.tokenizer.batch_decode(
                    queries,
                    skip_special_tokens=True,
                )

                try:
                    text_responses = self.tokenizer.batch_decode(
                        responses,
                        skip_special_tokens=False,
                    )
                except IndexError as e:
                    # This happens when the model generates an empty sequence.
                    # In this case, we just skip this sample.
                    raise ValueError(f"{responses.tolist()}")

                # Evaluate reward of the samples.
                text_responses = [
                    truncate_after_stop_token(text_response, self.tokenizer.eos_token)
                    for text_response in text_responses
                ]

                (
                    predicted_answers,
                    gt_answers,
                    levels,
                    symbolic_rewards,
                    outcome_metrics,
                ) = self._post_process_math_rollouts(text_responses, answer_gt_levels)

                rollouts_batch.update(outcome_metrics)

                # a penalty reward for not stopping.
                stop_token_penalty = self._stop_token_penalty(text_responses)

                rank0_print("=" * 20)
                rank0_print(
                    "Hyper-parameters:",
                    self.args.penalty_reward_value,
                    self.args.process_reward_upper_bound,
                    self.args.process_reward_scale,
                )
                rank0_print("Pred Answer:", predicted_answers[:8])
                rank0_print("GT Answer:", gt_answers[:8])
                rank0_print("Level:", levels[:8])
                rank0_print("Reward:", symbolic_rewards[:8])
                rank0_print("Stop Penalty:", stop_token_penalty[:8])
                rank0_print("=" * 20)

                rollouts_batch["text_queries"] = text_queries
                rollouts_batch["text_responses"] = text_responses

                responses = truncate_after_eos_padded(
                    responses, self.tokenizer.eos_id, self.tokenizer.pad_id
                )
                rollouts_batch["responses"] = responses
                rollouts_batch["stop_token_penalty"] = stop_token_penalty
                rollouts.append(rollouts_batch)

                del (
                    text_queries,
                    text_responses,
                    queries,
                    query_attn_masks,
                    responses,
                    stop_token_penalty,
                )  # To prevent mistakes.

        if self.args.compile:
            recover_all_backward_hooks(unwrapped_policy, all_backward_hooks)

        with torch.no_grad():
            for batch_idx, batch in tqdm.tqdm(
                enumerate(rollouts),
                total=len(rollouts),
                disable=not self.is_main_process,
                desc="compute_policy_outputs",
            ):
                # only policy and value models need mixed precision.
                with torch.cuda.amp.autocast(dtype=self.args.compute_dtype):
                    policy_outputs = self.sub_batched_outputs(
                        # model=self.policy,
                        model=unwrapped_policy,
                        rollouts_batch={
                            "queries": batch["queries"],
                            "query_attn_masks": batch["query_attn_masks"],
                            "responses": batch["responses"],
                        },
                        batch_size_per_device=batch["responses"].shape[0],
                        sub_batch_size=self.args.reward_model_per_device_batch_size,
                        temperature=self.args.temperature,
                    )

                policy_outputs = common_utils.unpack_dict(
                    policy_outputs,
                    keys=(
                        "logprobs",
                        "entropies",
                        "reg_entropies",
                        "reg_entropies_weight",
                    ),
                    return_type=dict,
                )
                policy_outputs["logprobs"] = policy_outputs["logprobs"].float()
                policy_outputs["entropies"] = policy_outputs["entropies"].float()
                if self.args.maxent_normalization == "full":
                    policy_outputs["reg_entropies"] = policy_outputs[
                        "reg_entropies"
                    ].float() / policy_outputs["reg_entropies_weight"].float().sum(
                        dim=-1, keepdim=True
                    )
                elif self.args.maxent_normalization == "none":
                    policy_outputs["reg_entropies"] = policy_outputs[
                        "reg_entropies"
                    ].float()
                elif self.args.maxent_normalization == "merged":
                    policy_outputs["reg_entropies"] = policy_outputs[
                        "reg_entropies"
                    ].float().sum(dim=-1) / policy_outputs[
                        "reg_entropies_weight"
                    ].float().sum(
                        dim=-1
                    )
                else:
                    raise ValueError(
                        f"Unknown maxent_normalization: {self.args.maxent_normalization}"
                    )
                batch.update(policy_outputs)

        with torch.no_grad():
            for batch_idx, batch in tqdm.tqdm(
                enumerate(rollouts),
                total=len(rollouts),
                disable=not self.is_main_process,
                desc="compute_value_outputs",
            ):
                # only policy and value models need mixed precision.
                with torch.cuda.amp.autocast(dtype=self.args.compute_dtype):
                    value_outputs = self.sub_batched_outputs(
                        model=self.value_model,
                        rollouts_batch={
                            "queries": batch["queries"],
                            "query_attn_masks": batch["query_attn_masks"],
                            "responses": batch["responses"],
                        },
                        batch_size_per_device=rollouts_batch["responses"].shape[0],
                        sub_batch_size=self.args.reward_model_per_device_batch_size,
                    )

                value_outputs = common_utils.unpack_dict(
                    value_outputs,
                    keys=("values",),
                    return_type=dict,
                )
                value_outputs["values"] = value_outputs["values"].float()
                batch.update(value_outputs)

        with torch.inference_mode():
            for batch_idx, batch in tqdm.tqdm(
                enumerate(rollouts),
                total=len(rollouts),
                disable=not self.is_main_process,
                desc="compute_ref_policy_outputs",
            ):
                # Evaluate logprobs of the samples.
                with torch.cuda.amp.autocast(dtype=self.args.compute_dtype):
                    ref_policy_outputs = self.sub_batched_outputs(
                        model=self.ref_policy,
                        rollouts_batch={
                            "queries": batch["queries"],
                            "query_attn_masks": batch["query_attn_masks"],
                            "responses": batch["responses"],
                        },
                        batch_size_per_device=batch["responses"].shape[0],
                        sub_batch_size=self.args.reward_model_per_device_batch_size,
                        temperature=self.args.temperature,
                    )

                ref_policy_outputs = common_utils.unpack_dict(
                    ref_policy_outputs, keys=("logprobs", "entropies"), return_type=dict
                )
                ref_policy_outputs["logprobs"] = ref_policy_outputs["logprobs"].float()
                ref_policy_outputs["entropies"] = ref_policy_outputs[
                    "entropies"
                ].float()
                batch.update(
                    {f"ref_{key}": value for key, value in ref_policy_outputs.items()}
                )

            for batch_idx, batch in tqdm.tqdm(
                enumerate(rollouts),
                total=len(rollouts),
                disable=not self.is_main_process,
                desc="compute_reward_outputs",
            ):
                stop_token_penalty = batch["stop_token_penalty"]
                text_queries = batch["text_queries"]
                text_responses = batch["text_responses"]
                del batch["stop_token_penalty"]
                del batch["text_queries"]
                del batch["text_responses"]

                sequences = torch.concat((batch["queries"], batch["responses"]), dim=1)

                padded_sequences, padding_shifts = rl_model.prepare_right_pad_sequences(
                    input_ids=sequences,
                    pad_token_id=self.tokenizer.pad_id,
                )

                with torch.cuda.amp.autocast(dtype=self.args.compute_dtype):
                    rewards = self.sub_batched_outputs(
                        model=self.reward_model,
                        rollouts_batch={"idx": padded_sequences},
                        batch_size_per_device=batch["responses"].size(0),
                        sub_batch_size=self.args.reward_model_per_device_batch_size,
                    )

                rewards = rl_model.restore_from_right_pad_sequences(
                    rewards, padding_shifts
                )
                rewards = rewards.float()

                encoded_full_seqs = sequences.tolist()
                scores = rewards.tolist()
                assert len(encoded_full_seqs) == len(scores)

                newline_id = self.tokenizer.encode("\n\n\n", bos=False, eos=False)[-1]

                packed_post_process = [
                    post_process_newline_scores(
                        encoded_full_seqs[i],
                        scores[i],
                        text_queries[i],
                        text_responses[i],
                        newline_id=newline_id,
                    )
                    for i in range(len(encoded_full_seqs))
                ]

                new_line_scores = [_[0] for _ in packed_post_process]
                beautiful_outputs = [_[1] for _ in packed_post_process]

                rank0_print("=" * 20)
                rank0_print("Reward:", beautiful_outputs[0])
                rank0_print("=" * 20)

                assert not (
                    self.args.apply_process_reward
                    and self.args.apply_terminal_process_reward
                ), "Cannot use both apply_process_reward and apply_terminal_process_reward."

                if self.args.maxent_normalization == "merged":
                    assert (
                        batch["symbolic_rewards"].shape == batch["reg_entropies"].shape
                    )
                    batch["symbolic_rewards"] = (
                        batch["symbolic_rewards"]
                        + batch["reg_entropies"] * self.args.maxent_coef
                    )
                    batch["reg_entropies"] = torch.zeros_like(
                        batch["reg_entropies"]
                    ).unsqueeze(dim=-1)

                (
                    shaped_rewards,
                    aggregated_rewards,
                    responses,
                ) = self.shape_process_rewards(
                    batch["symbolic_rewards"],
                    batch["queries"],
                    batch["responses"],
                    new_line_scores,
                    stop_token_penalty,
                )

                batch["responses"] = responses
                batch["rewards"] = aggregated_rewards

                # Shape reward with KL penalty.
                shape_reward_outputs = self._shape_reward(
                    rewards=(
                        shaped_rewards
                        if self.args.apply_process_reward
                        else aggregated_rewards
                    ),
                    responses=batch["responses"],
                    logprobs=batch["logprobs"],
                    ref_logprobs=batch["ref_logprobs"],
                    reg_entropies=batch["reg_entropies"],
                    terminal_reward=not self.args.apply_process_reward,
                )
                batch.update(shape_reward_outputs)

            cpu_rollouts = []
            for rollouts_batch in rollouts:
                rollouts_batch_cpu = {
                    key: value.cpu() for key, value in rollouts_batch.items()
                }
                cpu_rollouts.append(rollouts_batch_cpu)
            rollouts = cpu_rollouts

            # Items in dict need to be of same shape.
            rollouts = common_utils.merge_dict(rollouts, merge_fn=torch.cat)

            # Estimating advantages outside the loop gives more samples for reward normalization.
            advantages = self._estimate_advantage(
                rewards=rollouts["shaped_rewards"].to(self.device),
                values=rollouts["values"].to(self.device),
            )
            advantages = {key: value.cpu() for key, value in advantages.items()}

        return {**rollouts, **advantages}

    def post_terminating_reward(
        self,
        reward_outputs: Dict[str, torch.Tensor],
        responses: torch.Tensor,
        penalize_no_stop_token: bool,
        relative_stop_token_penalty: bool,
        stop_token_penalty: List[float],
    ) -> Dict[str, torch.Tensor]:
        """Assign bad reward values to sequences which didn't stop properly."""
        if penalize_no_stop_token:
            stop_token_penalty = torch.tensor(
                stop_token_penalty, device=responses.device
            )
            rewards = reward_outputs["rewards"]
            if relative_stop_token_penalty:
                rewards = rewards + stop_token_penalty * self.args.penalty_reward_value
            else:
                rewards[stop_token_penalty > 0.0] = self.args.penalty_reward_value
            reward_outputs["rewards"] = rewards
            return reward_outputs
        else:
            return reward_outputs

    def _post_process_math_rollouts(self, text_responses, answer_gt_levels):
        if self.args.stop_token is not None:
            parsed_stop_token = self.args.stop_token
            parsed_stop_token = parsed_stop_token.replace(r"\n", "\n")
            parsed_stop_token = parsed_stop_token.replace(r"\\", "\\")
        else:
            parsed_stop_token = self.tokenizer.eos_token

        predicted_answers = []
        for text_response in text_responses:
            predicted_answer = "No answer found."
            if "\n\n" in parsed_stop_token:
                if parsed_stop_token in text_response:
                    predicted_answer = text_response.split(parsed_stop_token)[1]
                    predicted_answer = predicted_answer.split(self.tokenizer.eos_token)[
                        0
                    ]
            elif "\\boxed{}" == parsed_stop_token:
                boxed_predicted_answer = text_response.split(self.tokenizer.eos_token)[
                    0
                ]
                boxed_predicted_answer = remove_boxed(
                    last_boxed_only_string(boxed_predicted_answer)
                )
                if boxed_predicted_answer is not None:
                    predicted_answer = boxed_predicted_answer
            else:
                raise ValueError(f"Unknown stop token: {parsed_stop_token}")
            predicted_answers.append(predicted_answer)

        text_answers_gt_levels = self.tokenizer.batch_decode(
            answer_gt_levels,
            skip_special_tokens=True,
        )

        answers, gt_answers, levels = [], [], []
        for text_answers_gt_level in text_answers_gt_levels:
            assert len(text_answers_gt_level.split(";;;")) == 3, text_answers_gt_level
            answer, gt_answer, level = text_answers_gt_level.split(";;;")
            answers.append(answer.strip())
            gt_answers.append(gt_answer.strip())
            levels.append(int(level.strip()))

        outcome_metrics, symbolic_rewards = self._calculate_outcome_accuracy(
            predicted_answers, gt_answers, answers, levels
        )
        return (
            predicted_answers,
            gt_answers,
            levels,
            symbolic_rewards,
            outcome_metrics,
        )

    def _stop_token_penalty(self, text_responses):
        if self.args.stop_token is not None:
            parsed_stop_token = self.args.stop_token
            parsed_stop_token = parsed_stop_token.replace(r"\n", "\n")
            parsed_stop_token = parsed_stop_token.replace(r"\\", "\\")
        else:
            parsed_stop_token = self.tokenizer.eos_token

        stop_token_penalty = []
        for text_response in text_responses:
            if parsed_stop_token not in text_response:
                stop_token_penalty.append(1.0)
            else:
                if "\n\n" in parsed_stop_token:
                    main_solution, rest_tokens = text_response.split(
                        parsed_stop_token, 1
                    )
                    main_solution = main_solution.strip()
                    rest_tokens = rest_tokens.split(self.tokenizer.eos_token)[0]

                    if self.args.minimal_reasoning_steps is not None:
                        if (
                            len(main_solution.split("\n\n"))
                            < self.args.minimal_reasoning_steps
                        ):
                            stop_token_penalty.append(1.0)
                            continue

                    if rest_tokens.strip() == "":
                        stop_token_penalty.append(1.0)
                        continue

                    if ("\\boxed{" + rest_tokens + "}") in main_solution:
                        # The best cast: The answer should be highlighted with \boxed{}.
                        stop_token_penalty.append(0.0)
                    elif (
                        f"${rest_tokens}$" in main_solution.split("\n\n")[-1]
                        or f"${rest_tokens}.$" in main_solution.split("\n\n")[-1]
                    ):
                        # The second best cast: The answer should be highlighted with math mode.
                        stop_token_penalty.append(0.5)
                    elif rest_tokens in main_solution.split("\n\n")[-1]:
                        # The third best cast: The answer should show up in the last line.
                        stop_token_penalty.append(0.75)
                    else:
                        stop_token_penalty.append(1.0)
                elif "\\boxed{}" == parsed_stop_token:
                    if "\\boxed" in text_response:
                        stop_token_penalty.append(0.0)
                    else:
                        stop_token_penalty.append(1.0)
                else:
                    raise ValueError(f"Unknown stop token: {parsed_stop_token}")
        return stop_token_penalty

    def shape_process_rewards(
        self,
        symbolic_rewards: torch.Tensor,
        queries: torch.Tensor,
        responses: torch.Tensor,
        new_line_scores: List[List[Tuple[int, float]]],
        stop_token_penalty: List[bool],
    ):
        assert (
            self.args.reward_aggregation == "min"
        ), "Only min is supported in reward aggregation for now."

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        responses_np = responses.cpu().numpy()
        for idx, per_solution_new_line_scores in enumerate(new_line_scores):
            for pos, score in per_solution_new_line_scores:
                assert pos >= 0, f"Not non-negative: {pos}"
                # # TODO (zhiqings): 13 is the id of newline token in llama
                # assert responses_np[idx, pos] == 13, f"Not 13: {responses_np[idx, pos]}"
                # assert (
                #     responses_np[idx, pos + 1] == 13
                # ), f"Not 13: {responses_np[idx, pos + 1]}"

        shaped_rewards = np.zeros((responses.size(0), responses.size(1)), dtype=float)

        for idx, per_solution_new_line_scores in enumerate(new_line_scores):
            scores = [sigmoid(_[1]) for _ in per_solution_new_line_scores]
            effetive_negative_step = -1
            if self.args.process_reward_scheme == "min":
                min_score = self.args.process_reward_upper_bound
                normalized_score = []
                for score in scores:
                    if score < min_score:
                        normalized_score.append(score - min_score)
                        min_score = score
                        effetive_negative_step = len(normalized_score)
                    else:
                        normalized_score.append(0.0)
            elif self.args.process_reward_scheme == "prod":
                prod_score = 1.0
                min_score = self.args.process_reward_upper_bound
                normalized_score = []
                for score in scores:
                    prod_score = prod_score * score
                    if prod_score < min_score:
                        normalized_score.append(prod_score - min_score)
                        min_score = prod_score
                        effetive_negative_step = len(normalized_score)
                    else:
                        normalized_score.append(0.0)
            else:
                raise ValueError(
                    f"Unknown process_reward_scheme: {self.args.process_reward_scheme}"
                )

            # TODO (zhiqings): 0.25 is a magic number.
            positive_score = 0.25
            per_step_positive_score = positive_score / (len(scores) + 1e-6)
            normalized_score = [_ + per_step_positive_score for _ in normalized_score]

            if self.args.truncate_on_negative_step:
                if effetive_negative_step != -1:
                    normalized_score = normalized_score[:effetive_negative_step]
                    per_solution_new_line_scores = per_solution_new_line_scores[
                        :effetive_negative_step
                    ]
                    last_pos = per_solution_new_line_scores[-1][0]
                    responses_np[idx, last_pos + 2 :] = 0
                    stop_token_penalty[idx] = 1.0
                elif stop_token_penalty[idx] == 1.0:
                    # we penalize the non-stop-token case.
                    stop_token_penalty[idx] = 2.0

            assert len(per_solution_new_line_scores) == len(normalized_score)
            for pos, score in zip(
                [_[0] for _ in per_solution_new_line_scores], normalized_score
            ):
                shaped_rewards[idx, pos] = score / 2.0
                shaped_rewards[idx, pos + 1] = score / 2.0

            if idx == 0:
                rank0_print("=" * 20)
                rank0_print("Normalized Scores:", normalized_score)
                rank0_print("=" * 20)

        if self.args.truncate_on_negative_step:
            responses = torch.tensor(responses_np, device=responses.device)

        terminal_rewards = symbolic_rewards.clone()

        terminal_rewards = self.post_terminating_reward(
            {"rewards": terminal_rewards},
            responses,
            penalize_no_stop_token=self.args.penalize_no_stop_token,
            relative_stop_token_penalty=self.args.relative_stop_token_penalty,
            stop_token_penalty=stop_token_penalty,
        )["rewards"]

        if self.args.apply_process_reward or self.args.apply_terminal_process_reward:
            shaped_rewards = torch.tensor(shaped_rewards, device=self.device)
            terminal_positions = (responses != self.tokenizer.pad_id).sum(dim=1) - 1
            shaped_rewards = shaped_rewards * self.args.process_reward_scale
            shaped_rewards[
                torch.arange(queries.size(0), device=responses.device),
                terminal_positions,
            ] += terminal_rewards

            if self.args.apply_process_reward:
                return shaped_rewards, shaped_rewards.sum(dim=1), responses
            else:
                return None, shaped_rewards.sum(dim=1), responses
        else:
            return None, terminal_rewards, responses

    def compute_policy_loss(
        self, rollouts: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        (
            old_logprob,
            advantages,
            queries,
            query_attn_masks,
            responses,
        ) = common_utils.prepare_inputs(
            common_utils.unpack_dict(
                rollouts,
                keys=(
                    "logprobs",
                    "advantages",
                    "queries",
                    "query_attn_masks",
                    "responses",
                ),
            ),
            device=self.device,
        )
        # Enable training mode for graident checkpointing.
        self.policy.train()

        outputs = self.policy(
            queries,
            query_attn_masks,
            responses,
            temperature=self.args.temperature,
        )

        logprob = outputs["logprobs"]

        with torch.autocast(device_type="cuda", enabled=False):
            advantages = advantages.float()
            logprob = logprob.float()
            old_logprob = old_logprob.float()

            ratio = torch.exp(logprob - old_logprob)
            # When current policy is close to the old policy, the KL component of this advantage is approximately correct.
            pg_losses = -advantages * ratio
            pg_losses2 = -advantages * torch.clamp(
                ratio, min=1.0 - self.args.cliprange, max=1.0 + self.args.cliprange
            )
            pg_loss = torch.maximum(pg_losses, pg_losses2).mean()
            pg_clipfrac = (
                (pg_losses2 > pg_losses).to(torch.get_default_dtype()).mean()
            )  # noqa
            entropy = outputs["entropies"].mean()
            reg_entropies = outputs["reg_entropies"].sum(dim=-1).mean()
            approxkl = 0.5 * ((logprob - old_logprob) ** 2.0).mean()

            if self.args.ent_reg_coef > 0.0:
                pg_loss = pg_loss - self.args.ent_reg_coef * reg_entropies

        stats = dict(
            loss=dict(policy=pg_loss),
            policy=dict(
                entropy=entropy,
                reg_entropy=reg_entropies,
                approxkl=approxkl,
                clipfrac=pg_clipfrac,
            ),
        )
        return pg_loss, common_utils.flatten_dict(
            stats, sep="/", postprocess_fn=lambda x: x.detach()
        )

    def compute_value_loss(
        self, rollouts: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        (
            values,
            returns,
            queries,
            query_attn_masks,
            responses,
        ) = common_utils.prepare_inputs(
            common_utils.unpack_dict(
                rollouts,
                keys=(
                    "values",
                    "returns",
                    "queries",
                    "query_attn_masks",
                    "responses",
                ),
            ),
            device=self.device,
        )
        # Enable training mode for graident checkpointing.
        self.value_model.train()

        outputs = self.value_model(
            queries,
            query_attn_masks,
            responses,
        )

        vpred = outputs["values"]

        with torch.autocast(device_type="cuda", enabled=False):
            vpred = vpred.float()
            returns = returns.float()
            values = values.float()

            vpredclipped = torch.clamp(
                vpred,
                min=values - self.args.cliprange_value,
                max=values + self.args.cliprange_value,
            )
            vf_losses1 = (vpred - returns) ** 2.0
            vf_losses2 = (vpredclipped - returns) ** 2.0
            vf_loss = 0.5 * torch.maximum(vf_losses1, vf_losses2).mean()
            loss = self.args.vf_coef * vf_loss

            vf_clipfrac = (vf_losses2 > vf_losses1).to(torch.get_default_dtype()).mean()
            value_mean, value_var = values.mean(), values.var(unbiased=False)
            return_mean, return_var = returns.mean(), returns.var(unbiased=False)

        stats = dict(
            loss=dict(value=vf_loss),
            val=dict(
                vpred=vpred.mean(),
                error=((vpred - returns) ** 2).mean(),
                clipfrac=vf_clipfrac,
                mean=value_mean,
                var=value_var,
            ),
            returns=dict(mean=return_mean, var=return_var),
        )
        return loss, common_utils.flatten_dict(
            stats, sep="/", postprocess_fn=lambda x: x.detach()
        )

    def record_step_stats(self, train_stats, rollouts, step_idx, **kwargs):
        kl = rollouts["kl"]
        kl_sum_seq, kl_avg_seq = kl.sum(dim=1).mean(dim=0), kl.mean()
        shaped_rewards = rollouts["shaped_rewards"].sum(dim=1).mean(dim=0)
        non_score_rewards = rollouts["non_score_rewards"].sum(dim=1).mean(dim=0)
        rewards = rollouts["rewards"].mean(dim=0)
        outcome_accuracy = rollouts["outcome_accuracy"].mean(dim=0)
        stats = {
            f"objective/kl_coef": kwargs["kl_coef"],
            f"objective/kl_sum_seq": kl_sum_seq,
            f"objective/kl_avg_seq": kl_avg_seq,
            f"objective/shaped_rewards": shaped_rewards,
            f"objective/non_score_rewards": non_score_rewards,
            f"objective/rewards": rewards,  # Original model reward.
            f"objective/outcome_accuracy": outcome_accuracy,
            f"objective/lr": self.policy_optimizer.param_groups[0]["lr"],
            f"objective/entropies": rollouts["entropies"].mean(),
            f"objective/ref_entropies": rollouts["ref_entropies"].mean(),
        }
        for level in range(9):
            if f"level_{level}_counts" in rollouts:
                level_counts = rollouts[f"level_{level}_counts"].sum()
                level_accuracy = rollouts[f"level_{level}_accuracy"].sum()
                # sync across all processes for higher accuracy.
                sync_vector = (
                    torch.cat([level_counts.view(1), level_accuracy.view(1)])
                    .view(-1)
                    .to(device=self.device)
                )
                dist.all_reduce(sync_vector)
                sync_vector = sync_vector.to(device=torch.device("cpu"))
                train_stats[f"outcome_accuracy/level_{level}"] = sync_vector[1] / (
                    sync_vector[0] + 1e-8
                )

        for k, v in train_stats.items():
            stats[f"ppo/{k}"] = v.mean(dim=0)
        stats = {
            key: value.item() if torch.is_tensor(value) else value
            for key, value in stats.items()
        }
        if self.is_main_process:
            if self.args.report_to == "wandb" and wandb is not None:
                wandb.log(stats, step=step_idx)
            else:
                # Just print to stdout.
                print(f"Step {step_idx}: {stats}")
            if self.args.save_dir is not None:
                # Store rollout data to disk to debug.
                rollouts_to_disk = {
                    key: self.tokenizer.batch_decode(
                        truncate_after_eos(
                            tensor,
                            self.tokenizer.eos_id,
                        ),
                        skip_special_tokens=False,
                    )
                    for key, tensor in common_utils.unpack_dict(
                        rollouts, keys=("queries", "responses"), return_type=dict
                    ).items()
                }

                rewards = [str(_) for _ in rollouts["rewards"].tolist()]
                rollouts_to_disk["rewards"] = rewards

                rollouts_to_disk = pd.DataFrame(rollouts_to_disk).to_dict(
                    orient="records"
                )
                rollout_log_dir = os.path.join(self.args.save_dir, "rollouts")
                os.makedirs(rollout_log_dir, exist_ok=True)
                with open(
                    os.path.join(rollout_log_dir, f"step_{step_idx}.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(rollouts_to_disk, f, indent=4)
        return stats

    @staticmethod
    def sub_batched_outputs(
        model: torch.nn.Module,
        rollouts_batch: Dict[str, torch.Tensor],
        batch_size_per_device: int,
        sub_batch_size: Optional[int] = None,
        **kwargs,
    ):
        if sub_batch_size is None or sub_batch_size == batch_size_per_device:
            all_outputs = model(
                **rollouts_batch,
                **kwargs,
            )
        else:
            assert batch_size_per_device % sub_batch_size == 0

            all_outputs_list = []
            for sub_batch_idx in range(batch_size_per_device // sub_batch_size):
                sub_batch = {
                    key: value[
                        sub_batch_idx
                        * sub_batch_size : (sub_batch_idx + 1)
                        * sub_batch_size
                    ]
                    for key, value in rollouts_batch.items()
                }
                sub_batch_outputs = model(
                    **sub_batch,
                    **kwargs,
                )
                all_outputs_list.append(sub_batch_outputs)

            if isinstance(all_outputs_list[0], dict):
                all_outputs = common_utils.merge_dict(
                    all_outputs_list, merge_fn=torch.cat
                )
            elif isinstance(all_outputs_list[0], torch.Tensor):
                all_outputs = torch.cat(all_outputs_list, dim=0)
            else:
                raise ValueError(
                    f"Unknown type of sub_batch_outputs: {type(all_outputs_list[0])}"
                )

            del sub_batch_outputs
            del all_outputs_list
            del sub_batch
        return all_outputs

    @torch.no_grad()
    def resume_training(self, checkpoint_dir: Path) -> int:
        (
            policy_resume_from_checkpoint,
            resume_epoch,
            resume_global_step,
        ) = get_latest_checkpoint_path(checkpoint_dir, prefix="policy_")

        value_resume_from_checkpoint, _, global_step = get_latest_checkpoint_path(
            checkpoint_dir, prefix="value_"
        )

        print(f"Resuming policy from checkpoint: {policy_resume_from_checkpoint}")
        print(f"Resuming value from checkpoint: {value_resume_from_checkpoint}")
        print(f"Resuming (epoch {resume_epoch}, global step {resume_global_step})")
        reume_dict = load_checkpoint(
            policy_resume_from_checkpoint,
            model=self.policy.base_model,
            optimizer=self.policy_optimizer,
            scheduler=self.policy_lr_scheduler,
            use_fsdp=self.args.policy_model_fsdp,
        )

        if "metrics" in reume_dict:
            self.best_metrics = reume_dict["metrics"]

        load_checkpoint(
            value_resume_from_checkpoint,
            model=self.value_model.base_model,
            optimizer=self.value_optimizer,
            scheduler=self.value_lr_scheduler,
            use_fsdp=self.args.value_model_fsdp,
        )
        return global_step

    @torch.no_grad()
    def evaluate(self, step_idx: int, temperature: float = 0.0):
        """Evaluate by generating sequences with test prefixes.

        FSDP compat: all devices should do the forward pass, since sharded params need to be summoned.
                     only write results in the main process.
        """
        eval_stats = {}
        test_stats = {}
        if self.eval_dataset is not None:
            query_data, query_attn_mask_data, answer_data = (
                self.eval_dataset.queries,
                self.eval_dataset.query_attn_masks,
                self.eval_dataset.answers,
            )

            eval_stats = self._evaluate(
                step_idx,
                query_data,
                query_attn_mask_data,
                answer_data,
                temperature=temperature,
                eval_prefix="evaluate",
            )

        if self.test_dataset is not None:
            query_data, query_attn_mask_data, answer_data = (
                self.test_dataset.queries,
                self.test_dataset.query_attn_masks,
                self.test_dataset.answers,
            )

            test_stats = self._evaluate(
                step_idx,
                query_data,
                query_attn_mask_data,
                answer_data,
                temperature=temperature,
                eval_prefix="test",
            )

        return {**eval_stats, **test_stats}

    def _evaluate(
        self,
        step_idx,
        query_data,
        query_attn_mask_data,
        answer_data,
        temperature,
        eval_prefix,
    ):
        if any(
            item is None for item in (query_data, query_attn_mask_data, answer_data)
        ):
            rank0_print("No evaluation data, skipping evaluation.")
            return

        # split data on data parallel dimension.
        dp_rank = get_data_parallel_rank()
        dp_world_size = get_data_parallel_world_size()
        eval_data_size = 1 + (len(query_data) - 1) // dp_world_size
        split_start_idx = eval_data_size * dp_rank
        split_end_idx = min(eval_data_size * (dp_rank + 1), len(query_data))

        query_data = query_data[split_start_idx:split_end_idx]
        query_attn_mask_data = query_attn_mask_data[split_start_idx:split_end_idx]
        answer_data = answer_data[split_start_idx:split_end_idx]
        eval_data_size = len(query_data)

        assert (
            self.args.per_device_eval_batch_size
            == self.args.rollout_per_device_batch_size
        ), "per_device_eval_batch_size should be equal to rollout_per_device_batch_size for now."
        per_device_batch_size = (
            self.args.per_device_eval_batch_size * get_model_parallel_world_size()
        )

        (
            query_data,
            query_attn_mask_data,
            answer_data,
        ) = common_utils.pad_inputs_on_batch(
            (query_data, query_attn_mask_data, answer_data),
            per_device_batch_size=per_device_batch_size,
        )

        assert query_data.size(0) % per_device_batch_size == 0, (
            f"query_data.size(0)={query_data.size(0)} should be divisible by "
            f"per_device_batch_size={per_device_batch_size}"
        )
        eval_iter_size = query_data.size(0) // per_device_batch_size

        # Start evaluation.
        unwrapped_policy: rl_model.Policy = self.policy.eval()
        if self.unwrapped_policy is not None:
            # No need to sync parameters sync it's already synced in the rollouts.
            unwrapped_policy: rl_model.Policy = self.unwrapped_policy.eval()

        if self.args.compile:
            all_backward_hooks = remove_all_backward_hooks(unwrapped_policy)

        eval_rollouts: List[Dict[str, torch.Tensor]] = []
        for batch_idx in tqdm.tqdm(
            range(eval_iter_size),
            total=eval_iter_size,
            disable=not self.is_main_process,
            desc=eval_prefix,
        ):
            # Sample rollouts.
            eval_start_idx = batch_idx * per_device_batch_size
            eval_end_idx = (batch_idx + 1) * per_device_batch_size

            queries = query_data[eval_start_idx:eval_end_idx]
            query_attn_masks = query_attn_mask_data[eval_start_idx:eval_end_idx]
            answer_gt_levels = answer_data[eval_start_idx:eval_end_idx]

            assert queries.size(0) == per_device_batch_size, (
                f"queries.size(0)={queries.size(0)} should be equal to "
                f"per_device_batch_size={per_device_batch_size}"
            )
            queries, query_attn_masks, answer_gt_levels = common_utils.prepare_inputs(
                (queries, query_attn_masks, answer_gt_levels),
                device=self.device,
            )

            with torch.cuda.amp.autocast(dtype=self.args.compute_dtype):
                responses = unwrapped_policy.respond(
                    queries, query_attn_masks, temperature=temperature
                )["responses"]

            text_queries = self.tokenizer.batch_decode(
                queries,
                skip_special_tokens=True,
            )

            try:
                text_responses = self.tokenizer.batch_decode(
                    responses,
                    skip_special_tokens=False,
                )
            except IndexError as e:
                # This happens when the model generates an empty sequence.
                # In this case, we just skip this sample.
                raise ValueError(f"{responses.tolist()}")

            # Evaluate reward of the samples.
            text_responses = [
                truncate_after_stop_token(text_response, self.tokenizer.eos_token)
                for text_response in text_responses
            ]

            eval_rollouts_batch = {}
            eval_rollouts_batch["text_queries"] = text_queries
            eval_rollouts_batch["text_responses"] = text_responses

            outcome_metrics = self._post_process_math_rollouts(
                text_responses, answer_gt_levels
            )[-1]

            eval_rollouts_batch.update(outcome_metrics)
            eval_rollouts.append(eval_rollouts_batch)

        if self.args.compile:
            recover_all_backward_hooks(unwrapped_policy, all_backward_hooks)

        cpu_eval_rollouts = []
        for eval_rollouts_batch in eval_rollouts:
            cpu_eval_rollouts.append(
                {
                    key: value.cpu() if torch.is_tensor(value) else value
                    for key, value in eval_rollouts_batch.items()
                }
            )
        eval_rollouts = cpu_eval_rollouts

        def merge_fn(tensor_or_list):
            if isinstance(tensor_or_list[0], list):
                return list(chain(*tensor_or_list))
            else:
                return torch.cat(tensor_or_list, dim=0)

        eval_rollouts = common_utils.merge_dict(eval_rollouts, merge_fn=merge_fn)

        filtered_eval_rollouts = {}
        for key, value in eval_rollouts.items():
            filtered_eval_rollouts[key] = value[:eval_data_size]
        eval_rollouts = filtered_eval_rollouts

        results = [
            {
                "text_query": text_query,
                "text_response": text_response,
            }
            for text_query, text_response in common_utils.zip_(
                eval_rollouts["text_queries"],
                eval_rollouts["text_responses"],
            )
        ]
        # save on rank 0.
        tp_rank = get_model_parallel_rank()
        tp_world_size = get_model_parallel_world_size()
        if tp_rank == 0:
            if self.args.save_dir is not None:
                evaluate_log_dir = os.path.join(self.args.save_dir, eval_prefix)
                if self.is_main_process and not os.path.exists(evaluate_log_dir):
                    os.makedirs(evaluate_log_dir, exist_ok=True)
                # wait in the dp group.
                dist.barrier(group=get_data_parallel_group())
                with open(
                    os.path.join(
                        evaluate_log_dir, f"eval_results_{step_idx}_rank_{dp_rank}.json"
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(results, f, indent=4)
            rank0_print(f"Saved evaluation results to {evaluate_log_dir}")

        rank0_print(
            f"End evaluation at step: {step_idx}. Processed {len(results)} examples"
        )

        eval_stats = {}
        overall_counts = 0.0
        overall_accuracy = 0.0
        for level in range(9):
            if f"level_{level}_counts" in eval_rollouts:
                level_counts = eval_rollouts[f"level_{level}_counts"].sum()
                level_accuracy = eval_rollouts[f"level_{level}_accuracy"].sum()
                overall_counts += level_counts
                overall_accuracy += level_accuracy
                # sync across all processes for higher accuracy.
                sync_vector = (
                    torch.cat([level_counts.view(1), level_accuracy.view(1)])
                    .view(-1)
                    .to(device=self.device)
                )
                dist.all_reduce(sync_vector)
                sync_vector = sync_vector.to(device=torch.device("cpu"))
                eval_stats[f"{eval_prefix}/accuracy_level_{level}"] = sync_vector[1] / (
                    sync_vector[0] + 1e-8
                )
                eval_stats[f"{eval_prefix}/counts_level_{level}"] = (
                    sync_vector[0] / tp_world_size
                )

        sync_vector = (
            torch.cat([overall_counts.view(1), overall_accuracy.view(1)])
            .view(-1)
            .to(device=self.device)
        )
        dist.all_reduce(sync_vector)
        sync_vector = sync_vector.to(device=torch.device("cpu"))
        eval_stats[f"{eval_prefix}/accuracy_overall"] = sync_vector[1] / (
            sync_vector[0] + 1e-8
        )
        eval_stats[f"{eval_prefix}/counts_overall"] = sync_vector[0] / tp_world_size

        # convert to float for logging.
        eval_stats = {
            key: value.item() if torch.is_tensor(value) else value
            for key, value in eval_stats.items()
        }

        if self.is_main_process:
            if self.args.report_to == "wandb" and wandb is not None:
                wandb.log(eval_stats, step=step_idx)
            else:
                # Just print to stdout.
                print(f"Step {step_idx} (Eval): {eval_stats}")
        return eval_stats

    @torch.no_grad()
    def save_model(
        self,
        step_idx: int,
        metrics: Optional[Dict[str, float]] = None,
    ):
        # Save the policy model.
        checkpoint_hook(
            args=self.args,
            model=self.policy.base_model,
            optimizer=self.policy_optimizer,
            scheduler=self.policy_lr_scheduler,
            epoch=None,
            global_step=step_idx,
            epoch_length=None,
            use_fsdp=self.args.policy_model_fsdp,
            prefix="policy_",
            metrics=metrics,
        )

        if not self.args.save_only_model:
            # Save the value model.
            checkpoint_hook(
                args=self.args,
                model=self.value_model.base_model,
                optimizer=self.value_optimizer,
                scheduler=self.value_lr_scheduler,
                epoch=None,
                global_step=step_idx,
                epoch_length=None,
                use_fsdp=self.args.value_model_fsdp,
                prefix="value_",
                metrics=metrics,
            )


def make_models(
    tokenizer: FakePreTrainedTokenizer,
    args: TrainingArguments,
    use_tp: bool = True,
    resume_from_checkpoint: Optional[str] = None,
) -> dict:
    device = torch.device("cuda")
    rank0_print("Resuming from checkpoint: ", [resume_from_checkpoint])
    set_global_compile_mode(False)

    if get_data_parallel_world_size() > 1:
        assert args.value_model_fsdp, "FSDP is required for DDP training."

    def post_init_processing(model, is_trainable, model_bits, use_fsdp, cpu_offload):
        if is_trainable:
            model = model.train()
        else:
            model = model.eval()

        backbone_model = model
        if isinstance(model, RewardModel):
            backbone_model = model.backbone_model

        with torch.device(device):
            max_batch_size = (
                max(
                    args.rollout_per_device_batch_size,
                    args.step_per_device_batch_size,
                )
                * get_model_parallel_world_size()
            )

            backbone_model.setup_caches(
                max_batch_size=max_batch_size,
                max_seq_length=backbone_model.config.block_size,
                kv_cache=False,
            )

        assert not (is_trainable and model_bits < 16), "Cannot train with int8 or int4."

        if model_bits < 16:
            assert model_bits == 8, "Only int8 is supported."
            assert not use_fsdp, "FSDP is not supported with int8."
            rank0_print("Quantizing model ...")
            from models.quantize import WeightOnlyInt8QuantHandler

            simple_quantizer = WeightOnlyInt8QuantHandler(model)
            model = simple_quantizer.convert_for_runtime_on_the_fly()
            model = model.to(device=device)
            model = model.eval()

        if cpu_offload:
            assert use_fsdp, "FSDP is required for CPU offload of params and grads."

        if use_fsdp:
            model = FSDP(
                module=model,
                process_group=get_data_parallel_group(),
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={
                        TransformerBlock,
                    },
                ),
                mixed_precision=MixedPrecision(
                    param_dtype=args.compute_dtype,
                    reduce_dtype=(
                        torch.float32
                        if get_data_parallel_world_size() >= 16
                        else args.compute_dtype
                    ),
                    keep_low_precision_grads=(args.optim_dtype != torch.float32),
                    buffer_dtype=args.compute_dtype,
                ),
                cpu_offload=(
                    CPUOffload(
                        offload_params=True,
                    )
                    if cpu_offload
                    else None
                ),
                sharding_strategy=(
                    ShardingStrategy.SHARD_GRAD_OP
                    if args.slow_cross_node_comm
                    else ShardingStrategy.FULL_SHARD
                ),
                forward_prefetch=True,
                limit_all_gathers=(not args.slow_cross_node_comm),
                sync_module_states=True,
            )
        elif get_data_parallel_world_size() > 1 and is_trainable:
            model = DDP(
                module=model,
                device_ids=[device],
                broadcast_buffers=False,
                process_group=get_data_parallel_group(),
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )

        return model

    def make_generative_policy(
        is_trainable,
        base_path,
        resume_path=None,
        model_bits=16,
        use_fsdp=False,
        cpu_offload=False,
    ):
        model = load_model_from_from_ckpt(
            checkpoint_path=base_path,
            sft_checkpoint_path=resume_path,
            device=device,
            precision=(args.param_dtype if is_trainable else args.compute_dtype),
            use_tp=use_tp,
            requires_grad=is_trainable,
            skip_init=get_data_parallel_rank() > 0,
            vocab_parallel=args.vocab_parallel,
            sequence_parallel=args.sequence_parallel and is_trainable,
        )
        model = post_init_processing(
            model, is_trainable, model_bits, use_fsdp, cpu_offload
        )
        return model

    def make_reward_model(
        is_trainable,
        base_path,
        resume_path=None,
        backbone_only=False,
        model_bits=16,
        use_fsdp=False,
        cpu_offload=False,
    ):
        model = load_reward_model_from_ckpt(
            checkpoint_path=base_path,
            rm_checkpoint_path=resume_path,
            device=device,
            precision=(args.param_dtype if is_trainable else args.compute_dtype),
            use_tp=use_tp,
            requires_grad=is_trainable,
            skip_init=get_data_parallel_rank() > 0,
            vocab_parallel=args.vocab_parallel,
            sequence_parallel=args.sequence_parallel and is_trainable,
        )

        if backbone_only:
            model = model.backbone_model

        model = post_init_processing(
            model, is_trainable, model_bits, use_fsdp, cpu_offload
        )
        return model

    # Model construction below seems convoluted, but it's made to trade time for RAM efficiency.
    # For large models, object creation could be extremely RAM intensive.
    # Especially so for multiple processes on single node, each starting off with a copy of the model.
    # General strategy is to 1) create a model, 2) move it to target device / shard it, 3) then start next model,
    # as opposed to creating all needed models on CPU first, and separately moving / sharding each.

    rank0_print("### Creating the referecnce policy ...")
    ref_policy = rl_model.make_policy_with_base_model(
        args,
        make_generative_policy(
            is_trainable=False,
            base_path=args.base_checkpoint_path,
            resume_path=args.policy_checkpoint_path,
            model_bits=args.ref_policy_model_bits,
            use_fsdp=args.ref_policy_model_fsdp,
            cpu_offload=args.ref_policy_model_cpu_offload,
        ),
        tokenizer,
    )

    rank0_print("### Creating the policy ...")
    policy = rl_model.make_policy_with_base_model(
        args,
        make_generative_policy(
            is_trainable=True,
            base_path=args.base_checkpoint_path,
            resume_path=(
                args.policy_checkpoint_path if resume_from_checkpoint is None else None
            ),  # resuming policy from training is handled by the Trainer.
            use_fsdp=args.policy_model_fsdp,
            cpu_offload=args.policy_model_cpu_offload,
        ),
        tokenizer,
    )

    unwrapped_policy = None
    if args.policy_model_fsdp:
        rank0_print("### Creating the unwrapped policy ...")
        unwrapped_policy = rl_model.make_policy_with_base_model(
            args,
            make_generative_policy(
                is_trainable=False,
                base_path=args.base_checkpoint_path,
                resume_path=None,  # unwrapped_policy would be copied from policy.
            ),
            tokenizer,
        )

    rank0_print("### Creating the value model ...")
    if args.init_value_with_reward:
        # Initialize value from reward model a la OAI.
        rank0_print("Initializing value model with reward model.")
        value_model = rl_model.make_value_with_base_model(
            args,
            make_reward_model(
                is_trainable=True,
                base_path=(
                    args.value_base_checkpoint_path
                    or args.reward_base_checkpoint_path
                    or args.base_checkpoint_path
                ),
                resume_path=(
                    args.value_checkpoint_path or args.reward_checkpoint_path
                    if resume_from_checkpoint is None
                    else None
                ),
                backbone_only=True,
                use_fsdp=args.value_model_fsdp,
                cpu_offload=args.value_model_cpu_offload,
            ),
            tokenizer,
        )
    else:
        rank0_print("Initializing value model with policy model.")
        # Initialize value from policy. Works for sanity, but generally performs worse in instruction-following.
        value_model = rl_model.make_value_with_base_model(
            args,
            make_generative_policy(
                is_trainable=True,
                base_path=(
                    args.value_base_checkpoint_path or args.base_checkpoint_path
                ),
                resume_path=(
                    args.value_checkpoint_path or args.policy_checkpoint_path
                    if resume_from_checkpoint is None
                    else None
                ),
                use_fsdp=args.value_model_fsdp,
                cpu_offload=args.value_model_cpu_offload,
            ),
            tokenizer,
        )
        apply_reward_modeling_head(value_model.base_model, requires_grad=True)
        apply_reward_head_tp(value_model.base_model, requires_grad=True)

    rank0_print("### Creating the reward model ...")
    reward_model = make_reward_model(
        is_trainable=False,
        base_path=args.reward_base_checkpoint_path or args.base_checkpoint_path,
        resume_path=args.reward_checkpoint_path,
        model_bits=args.reward_model_bits,
        use_fsdp=args.reward_model_fsdp,
        cpu_offload=args.reward_model_cpu_offload,
    )

    return dict(
        policy=policy,
        value_model=value_model,
        ref_policy=ref_policy,
        reward_model=reward_model,
        unwrapped_policy=unwrapped_policy,
    )


def whiten(
    values: torch.Tensor, shift_mean=True, epsilon=1e-8, async_stats="full_batch"
) -> torch.Tensor:
    assert async_stats in ["full_batch", "per_gpu", "none"]

    values_for_statistics = values
    if async_stats == "full_batch":
        if not values_for_statistics.is_cuda:
            raise ValueError("SyncWhiten expected input tensor to be on GPU")

        need_sync = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )

        if need_sync:
            process_group = torch.distributed.group.WORLD
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        if need_sync:
            tensor_list = [
                torch.zeros_like(values_for_statistics) for _ in range(world_size)
            ]
            torch.distributed.all_gather(tensor_list, values_for_statistics)
            values_for_statistics = torch.cat(tensor_list, dim=0)

    if async_stats in ["full_batch", "per_gpu"]:
        assert (
            values_for_statistics.size(0) >= 8
        ), f"Internal error: Minibatch size {values.size(0)} is insufficient for whitening."
        mean = values_for_statistics.mean()  # noqa
        std = values_for_statistics.std(unbiased=False)  # noqa

    else:
        mean = values.mean(dim=-1, keepdim=True)
        std = values.std(dim=-1, unbiased=False, keepdim=True)

    whitened = (values - mean) / (std + epsilon)
    if not shift_mean:
        whitened = whitened + mean
    return whitened


def truncate_after_stop_token(
    response: str,
    stop_token: Optional[str] = None,
    additional_stop_token: Optional[List[str]] = None,
) -> str:
    if stop_token is None:
        return response

    if additional_stop_token is None:
        additional_stop_token = ["<s>", "</s>"]

    for token in additional_stop_token + [stop_token]:
        if len(response.split(token)) > 1:
            response = response.split(token)[0] + token

    return response


def rank0_print(*args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)


def remove_all_backward_hooks(model: torch.nn.Module) -> Dict[str, OrderedDict]:
    all_backward_hooks = {}

    for name, module in model.named_modules():
        all_backward_hooks[name] = module._backward_hooks
        module._backward_hooks = OrderedDict()

    return all_backward_hooks


def recover_all_backward_hooks(
    model: torch.nn.Module, all_backward_hooks: Dict[str, OrderedDict]
):
    for name, module in model.named_modules():
        if name in all_backward_hooks:
            module._backward_hooks = all_backward_hooks[name]


def post_process_newline_scores(
    encoded_full_seq,
    scores,
    prompt,
    output,
    str_splitter="\n\n",
    newline_id=13,
):
    if str_splitter == "\n\n":
        token_splitter = f" {newline_id}, {newline_id},"
    else:
        raise ValueError(f"Unknown str_splitter: {str_splitter}")
    encoded_full_seq_string = " " + ", ".join(str(_) for _ in encoded_full_seq) + ","

    prompt_newlines = prompt.count(str_splitter)
    output_newlines = output.count(str_splitter)
    full_seq_newlines = encoded_full_seq_string.count(token_splitter)

    # we get scores at the position from the end of the prompt
    # to the end of the output

    # if not prompt.endswith(str_splitter):
    #     newline_scores = []
    #     beautiful_output = [prompt.strip()]
    #     for segment in output.split(str_splitter):
    #         beautiful_output.append(segment.strip())
    #     return newline_scores, str_splitter.join(beautiful_output)
    assert prompt_newlines + output_newlines <= full_seq_newlines, (
        f"Something wrong: {prompt_newlines} + {output_newlines} v.s. {full_seq_newlines}\n\n"
        f"===\n{prompt}\n===\n{output}\n===\n{encoded_full_seq_string}\n==="
    )

    encoded_prompt_string = token_splitter.join(
        encoded_full_seq_string.split(token_splitter, prompt_newlines)[:-1]
    )
    prompt_seq_len = (
        len([_.strip() for _ in encoded_prompt_string.split(",") if _.strip()]) + 2
    )
    encoded_output_string = encoded_full_seq_string.split(
        token_splitter, prompt_newlines
    )[-1]

    splitted_encoded_full_seq_string = [
        _.strip() for _ in encoded_full_seq_string.split(",") if _.strip()
    ]

    newline_scores = []
    beautiful_output = [prompt.strip()]
    pos = prompt_seq_len
    assert (
        splitted_encoded_full_seq_string[pos - 1] == f"{newline_id}"
    ), f"Something wrong: {splitted_encoded_full_seq_string[pos - 3: pos + 2]} v.s. {newline_id}"
    assert (
        splitted_encoded_full_seq_string[pos - 2] == f"{newline_id}"
    ), f"Something wrong: {splitted_encoded_full_seq_string[pos - 4: pos + 1]} v.s. {newline_id}"

    # assert len(encoded_output_string.split(token_splitter)) == len(
    #     output.split(str_splitter)
    # ), (
    #     f"Something wrong: {len(encoded_output_string.split(token_splitter))}"
    #     f" v.s. {len(output.split(str_splitter))}\n"
    #     f"{encoded_output_string}"
    #     f"\n========{output}\n========"
    # )

    if len(encoded_output_string.split(token_splitter)) != len(
        output.split(str_splitter)
    ):
        newline_scores = []
        beautiful_output = "Parsing error"
        return newline_scores, beautiful_output

    for i, encoded_segment, segment in zip(
        range(output_newlines + 1),
        encoded_output_string.split(token_splitter),
        output.split(str_splitter),
    ):
        pos += len([_.strip() for _ in encoded_segment.split(",") if _.strip()]) + 2
        if i < output_newlines - 1:
            assert splitted_encoded_full_seq_string[pos - 1] == f"{newline_id}", (
                f"Something wrong: {splitted_encoded_full_seq_string[pos - 3: pos + 2]}"
                f" v.s. {newline_id}"
            )
            assert splitted_encoded_full_seq_string[pos - 2] == f"{newline_id}", (
                f"Something wrong: {splitted_encoded_full_seq_string[pos - 4: pos + 1]}"
                f" v.s. {newline_id}"
            )
            if pos < prompt_seq_len + 4:  # 4 is magic number
                newline_scores.append(
                    (
                        pos - 2 - prompt_seq_len,
                        -4.44,
                    )
                )  # noqa
                beautiful_output.append(f"{segment} (-4.44)")
            else:
                newline_scores.append(
                    (
                        pos - 2 - prompt_seq_len,
                        min(
                            scores[pos - 3],
                            scores[pos - 2],
                            scores[pos - 1],
                        ),
                    )
                )  # noqa
                beautiful_output.append(
                    f"{segment} ({scores[pos - 3]},{scores[pos - 2]},{scores[pos - 1]})"
                )
        else:
            beautiful_output.append(segment)

    return newline_scores, str_splitter.join(beautiful_output)


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None

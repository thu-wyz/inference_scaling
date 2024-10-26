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

import logging
from typing import Optional, Dict, Sequence

import torch
from torch.utils.data import Dataset

from datasets import Dataset as HFDataset

import data_utils.common_utils as utils
from data_utils.tokenizer_utils import FakePreTrainedTokenizer
from data_utils.data_utils_dpo import preprocess_for_dpo
from training_utils.training_args import TrainingArguments


logger = logging.getLogger(__name__)


DROMEDARY_PROMPT_DICT = {
    "prompt_input": (
        "{meta_prompt}\n" "{instruction}\n\n" "{input}\n\n" "### Dromedary"
    ),
    "prompt_no_input": ("{meta_prompt}\n" "{instruction}\n\n" "### Dromedary"),
}


# https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/alpaca_farm_greedy_gpt4/chatml_b5_without_inputs.txt
ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "You are a helpful assistant whose goal is to select the preferred (least wrong) output for a given instruction. "
        "Print 1 or 2 to indicate which output is better.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}"
    ),
    "prompt_no_input": (
        "You are a helpful assistant whose goal is to select the preferred (least wrong) output for a given instruction. "
        "Print 1 or 2 to indicate which output is better.\n\n"
        "### Instruction:\n{instruction}"
    ),
}


def format_prompt(
    example: Dict[str, str],
    prompt_dict: Dict[str, str],
) -> str:
    if prompt_dict is not None:
        assert (
            "instruction" in example
        ), "Internal error: example missing required keys."

        if example.get("input", "") != "":
            prompt_format = prompt_dict["prompt_input"]
        else:
            prompt_format = prompt_dict["prompt_no_input"]
    else:
        prompt_format = "{input}"

    format_prompt = prompt_format.format(**example)
    return format_prompt


def format_double_outputs(
    example: dict,
    output1_key="output_1",
    output2_key="output_2",
) -> str:
    return (
        f"\n\n### Output 1:\n{example[output1_key]}"
        f"\n\n### Output 2:\n{example[output2_key]}"
        f"\n\n### Preferred Output is "
    )


def _tokenize_fn(
    strings: Sequence[str],
    tokenizer: FakePreTrainedTokenizer,
    max_length: int,
    use_data_frame: bool = False,
) -> dict:
    """Tokenize a list of strings."""
    if use_data_frame:
        raise NotImplementedError
    strings_ds = strings

    tokenized_strings = tokenizer(
        strings_ds,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        add_bos=True,
        add_eos=False,  # reward model doesn't need eos
        padding_side="right",
        truncation_side="right",
    )

    input_ids = torch.tensor(tokenized_strings["input_ids"])
    return input_ids


def preprocess_for_preference_modeling(
    data: HFDataset,
    tokenizer: FakePreTrainedTokenizer,
    max_length: Optional[int] = None,
    query_len: Optional[int] = None,
    response_len: Optional[int] = None,
    prompt_dict: Optional[Dict[str, str]] = None,
) -> Dict[str, torch.Tensor]:
    list_dict_data = data.to_pandas().to_dict("records")

    def _get_numeric_preference(example: dict):
        # 1 vs 2 is stored in table, but for modeling we use 0 vs 1; remap here.
        return {1: 0, 2: 1, -1: -1}[example["preference"]]

    choice = torch.tensor(
        [[_get_numeric_preference(dict_data)] for dict_data in list_dict_data]
    )

    def _get_text(example: dict, output1_key: str, output2_key: str):
        full_prompt = format_prompt(example, prompt_dict) + format_double_outputs(
            example, output1_key, output2_key
        )
        return full_prompt

    text_list = [
        _get_text(dict_data, "output_1", "output_2") for dict_data in list_dict_data
    ]

    if max_length is None:
        max_length = query_len + response_len

    logger.warning(f"Tokenizing {len(list_dict_data)} pairs...")

    # "size" (bsz, seq_len)
    input_ids = _tokenize_fn(text_list, tokenizer, max_length)

    packaged_data = dict(
        input_ids=input_ids,
        choice=choice,
        metadata=dict(mean_choice=choice.float().mean().item()),
    )

    return packaged_data


class PairwisePreferenceModelingDataset(Dataset):
    def __init__(
        self,
        args: TrainingArguments,
        data: HFDataset,
        tokenizer: FakePreTrainedTokenizer,
        max_length: Optional[int] = None,
        query_len: Optional[int] = None,
        response_len: Optional[int] = None,
        prompt_dict: Optional[Dict[str, str]] = None,
    ):
        super(PairwisePreferenceModelingDataset, self).__init__()
        data_dict = preprocess_for_preference_modeling(
            data=data,
            tokenizer=tokenizer,
            max_length=max_length,
            query_len=query_len,
            response_len=response_len,
            prompt_dict=prompt_dict,
        )
        dpo_data_dict = preprocess_for_dpo(
            args=args,
            dataset=data,
            tokenizer=tokenizer,
            reorder_wl=False,
        )
        self.input_ids = data_dict["input_ids"]
        self.choice = data_dict["choice"]
        self.metadata = data_dict["metadata"]

        self.input_ids_1 = dpo_data_dict["input_ids_w"]
        self.input_ids_2 = dpo_data_dict["input_ids_l"]
        self.labels_1 = dpo_data_dict["labels_w"]
        self.labels_2 = dpo_data_dict["labels_l"]
        assert len(self.input_ids) == len(self.input_ids_1)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            choice=self.choice[i],
            input_ids_1=self.input_ids_1[i],
            labels_1=self.labels_1[i],
            input_ids_2=self.input_ids_2[i],
            labels_2=self.labels_2[i],
        )


def make_pairwise_preference_modeling_data_module(
    tokenizer: FakePreTrainedTokenizer,
    args: TrainingArguments,
    dpo_pm_total_max_len: Optional[int] = None,
):
    preference_dataset = utils.local_dataset(args.dataset)
    train_preference = preference_dataset["train"]

    if args.dataset_format == "alpaca":
        prompt_dict = ALPACA_PROMPT_DICT
    elif args.dataset_format is None:
        prompt_dict = None
    else:
        raise ValueError(
            f"Unsupported dataset_format: {args.dataset_format}."
            "Only alpaca and None are supported."
        )

    train_dataset = PairwisePreferenceModelingDataset(
        args=args,
        data=train_preference,
        tokenizer=tokenizer,
        max_length=dpo_pm_total_max_len or args.total_max_len,
        query_len=args.source_max_len,
        response_len=args.target_max_len,
        prompt_dict=prompt_dict,
    )

    eval_dataset = None
    if args.eval_size > 0:
        train_dataset, eval_dataset = utils.split_train_into_train_and_eval(
            train_dataset=train_dataset,
            eval_size=args.eval_size,
            seed=args.seed,
        )

    data_collator = utils.DataCollatorForStackableDataset()
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

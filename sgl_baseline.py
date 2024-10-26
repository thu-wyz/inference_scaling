import argparse
import json
import re
import time
from sglang import function, gen, RuntimeEndpoint
import math
import threading
from evaluate.data_processing.answer_extraction import *
from evaluate.evaluate_utils.grader import *


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_prompts_math(args):
    test_cases = read_jsonl(args.input_path)
    prompts = []
    for test in test_cases:
        prompts.append(test["problem"])
    return prompts, test_cases

def get_prompts_gsm(args):
    examples = read_jsonl(args.input_path)
    prompts = []

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")
        prompts.append(ex["question"])

    return prompts, examples

@function
def multiple_sampling(s, id, question, sampling_num, max_tokens, ground_truth_answer, temperature, model_type):
    if model_type == "metamath":
        prompt = "# Question\n\n" + question + "\n\n# Solution"
        s += "# Question\n\n" + question + "\n\n# Solution"
        question = prompt
    if model_type == "mistral_7b" or "llemma" in model_type:
        s += question
    forks = s.fork(sampling_num)
    forks += gen("answer", max_tokens=max_tokens, temperature=temperature)
    answers = []
    total_tokens = 0
    for state in forks:
        text = state.text()
        num_gen_tokens = state.get_meta_info("answer")["completion_tokens"]
        total_tokens += num_gen_tokens
        answers.append({'text':text, "num_gen_tokens":num_gen_tokens})
    answer_for_the_question = {"id":id, "question": question, "model_answer":answers, "ground_truth_answer": ground_truth_answer["answer"], "total_tokens":total_tokens}
    return answer_for_the_question




def main(args):
    if args.dataset == "gsm8k":
        prompts, test_examples = get_prompts_gsm(args)
    else:
        prompts, test_examples = get_prompts_math(args)
    input_list_dict = []
    for i, prompt in enumerate(prompts):
        input_list_dict.append({"id": i, "question": prompt, "sampling_num": args.sampling_num, "max_tokens": args.max_tokens, "ground_truth_answer": test_examples[i], "temperature": args.temperature, "model_type": args.model_type})
    states = multiple_sampling.run_batch(input_list_dict, backend=RuntimeEndpoint(args.policy_host), num_threads=args.num_threads, progress_bar=True)
    results = []
    total_gen_tokens = 0
    for s, truth in zip(states, test_examples):
        answer = s.ret_value
        total_gen_tokens += answer["total_tokens"]
        results.append(answer)
    json.dump(results, open(args.output_path, "w"), indent=4)



if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--input_path', type=str, required=True)
    args_parser.add_argument('--output_path', type=str, required=True)
    args_parser.add_argument('--sampling_num', type=int, default=32)
    args_parser.add_argument('--max_tokens', type=int, default=1024)
    args_parser.add_argument('--policy_host', type=str, default="http://localhost:30100")
    args_parser.add_argument('--num_threads',  type=int, required=True)
    args_parser.add_argument('--temperature', type=float, default=1.0)
    args_parser.add_argument('--model_type', type=str, choices=["mistral_7b", "llemma_7b", "llemma_34b", "metamath"])
    args_parser.add_argument('--dataset', type=str)
    args = args_parser.parse_args()
    main(args)
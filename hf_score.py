from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
import argparse
import json
import tqdm
from evaluate.data_processing.answer_extraction import *

step_tag = 'ки'

COT = "\nPlease reason step by step, and put your final answer within \\boxed{}."
good_token = '+'
bad_token = '-'

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    step_tag_id = tokenizer.encode(f" {step_tag}")[-1]
    dev = torch.device(f"cuda:{args.dev}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).eval()
    model = model.to(dev)
    intput_data = json.load(open(args.input_path, "r"))
    output_data_list = []
    for data in tqdm.tqdm(intput_data):
        prompts = []
        for answer in data["model_answer"]:
            text = answer["text"]
            if args.shepherd == False:
                qa = text.split(COT)
                question = qa[0]
                solution = qa[1]
                solution = solution.replace(".\n", ".\n ки ")
                solution = solution.replace("\\]", "\\] ки ")
                solution = solution.replace("$\n", "$\n ки ")
                text = question + COT + solution + " ки " 
            prompts.append(text)
        pos = 0
        while pos < len(prompts):
            encoded_input = tokenizer(prompts[pos:pos+args.batch_size], padding=True, truncation=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                scores = model(**encoded_input).logits.mean(dim=-1).sigmoid()
                for i, step_scores in enumerate(scores):
                    step_scores = step_scores[encoded_input["input_ids"][i] == step_tag_id]
                    step_scores = step_scores.cpu().tolist()
                    data["model_answer"][pos+i]["step_scores"] = step_scores
                data["ground_truth_answer"] = extract_answer(data["ground_truth_answer"])
            pos += args.batch_size
        output_data_list.append(data)
    json.dump(output_data_list, open(args.output_path, "w"), indent=4)



if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--input_path', type=str, required=True)
    args_parser.add_argument('--output_path', type=str, required=True)
    args_parser.add_argument('--batch_size', type=int, required=True)
    args_parser.add_argument('--dev', type=str, required=True)
    args_parser.add_argument('--model_path', type=str, required=True)
    args_parser.add_argument('--shepherd', type=bool, default=False)
    args = args_parser.parse_args()
    main(args)
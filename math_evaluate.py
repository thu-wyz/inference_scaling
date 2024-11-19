import os 
import re
import json
import tqdm
import argparse
import sys
import os
import torch
from evaluate.evaluate_utils.grader import *
from collections import Counter
from evaluate.data_processing.answer_extraction import extract_answer

ANS_RE = re.compile(r"The answer is: (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
GT_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
SHEPHERD_RE = re.compile(r"The answer is:(.+) \u043a\u0438")
METAMATH_RE = re.compile(r"The answer is:(.+)\n\n")


def extract_gsm_answer(completion):
    match = ANS_RE.search(completion)
    if match: 
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def extract_shepherd_answer(completion):
    if completion == None:
        return INVALID_ANS
    match = SHEPHERD_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS



def extract_metamath_answer(completion):
    match = METAMATH_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def extract_ground_truth(completion):
    match = GT_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str


def agg_min(step_scores):
    if step_scores == []:
        return 0
    minn = 1
    for s in step_scores:
        if s is not None:
            if s < minn:
                minn = s
    return minn

def agg_mean(step_scores):
    if step_scores == []:
        return 0
    return sum(step_scores) / len(step_scores)

def agg_prod(step_scores):
    prod = 1.0
    if step_scores == []:
        return 0
    for score in step_scores:
        prod *= score
    return prod

def agg_last(step_scores):
    if step_scores == []:
        return 0
    if step_scores[-1] == None:
        return 0
    return step_scores[-1]

def evaluate(path, aggfunc, extract_function):
    data = json.load(open(path, "r"))
    num_correct = 0
    data = data
    total = len(data)
    for qapair in data:
        max_score = -999.999
        answer = None
        for cand in qapair["model_answer"]:
            if aggfunc(cand["step_scores"]) > max_score:
                answer = cand["text"]
                max_score = aggfunc(cand["step_scores"])
        answer = extract_function(answer)
        ground_truth = qapair["ground_truth_answer"]
        if grade_answer(answer, ground_truth):
            num_correct += 1
    print(num_correct)
    return num_correct * 1.0 / total

def majority_vote(path, weighted, weight_func, extract_function):
    data = json.load(open(path, "r"))
    num_correct = 0
    total = len(data)
    for qapair in data:
        max_vote = 0
        max_rep = None
        equiv_classes = []
        equiv_weights = []
        for cand in qapair["model_answer"]:
            answer = extract_function(cand["text"])
            weight = 1
            if weighted == True:
                weight = weight_func(cand["step_scores"])
            flag = 0
            for i, rep in enumerate(equiv_classes):
                if grade_answer(answer,rep):
                    flag = 1
                    equiv_weights[i] = equiv_weights[i]+weight
                    if equiv_weights[i] > max_vote:
                        max_vote = equiv_weights[i]
                        max_rep = answer
            if flag:
                continue
            equiv_classes.append(answer)
            equiv_weights.append(weight)
            if weight > max_vote:
                max_vote = weight
                max_rep = answer
        if grade_answer(str(max_rep), qapair["ground_truth_answer"]):
            num_correct += 1
    print(num_correct)
    print(num_correct * 1.0 / total)
    return num_correct * 1.0 / total



def main(args):
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    sys.path.append(os.path.join(current_dir, 'eval_deepseek')) 
    if args.model_type == "mistral_7b" or args.model_type == "llemma":
        extract_func = extract_shepherd_answer
    if args.agg_func == "min":
        aggfunc = agg_min
        accuracy = evaluate(args.path, aggfunc, extract_func)
    if args.agg_func == "prod":
        aggfunc = agg_prod
        accuracy = evaluate(args.path, aggfunc, extract_func)
    if args.agg_func == "mean":
        aggfunc = agg_mean
        accuracy = evaluate(args.path, aggfunc, extract_func)
    if args.agg_func == "last":
        aggfunc = agg_last
        accuracy = evaluate(args.path, aggfunc, extract_func)
    if args.agg_func == "majority_vote":
        weight_func = None
        if args.weight_agg == "min":
            weight_func = agg_min
        if args.weight_agg == "prod":
            weight_func = agg_prod
        if args.weight_agg == "mean":
            weight_func = agg_mean
        if args.weight_agg == "last":
            weight_func = agg_last
        accuracy = majority_vote(args.path, args.weighted, weight_func, extract_func)
    if args.output_path:
        with open(args.output_path, "a") as file:
            file.write(f"\nagg_function {args.agg_func} weighted: {args.weighted} \n accuracy: {str(accuracy)}")
    return accuracy


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--path', type=str, required=True)
    args_parser.add_argument('--agg_func', type=str, choices=["min", "prod", "mean", "last", "majority_vote"])
    args_parser.add_argument('--weighted', type=bool, default=False)
    args_parser.add_argument('--weight_agg', type=str, choices=["min", "prod", "mean", "last"])
    args_parser.add_argument('--model_type', type=str, choices=["mistral", "llemma"])
    args_parser.add_argument('--output_path', type=str)
    args = args_parser.parse_args()
    main(args)










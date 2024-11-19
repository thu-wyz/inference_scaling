# Code Repo
[**Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models**](https://arxiv.org/abs/2408.00724).

## Clone
    git clone --recurse-submodules git@github.com:thu-wyz/inference_scaling.git
This command will clone our repository with the [sglang](https://github.com/sgl-project/sglang) repository as a submodule. The sglang repository should be on the *reward-model* branch, which has been modified slightly by us to support our process reward model for efficient tree search.
One can also use hf_score.py in the repo to score the steps of each solution.
The benchmark datasets: [MATH](https://github.com/hendrycks/math), [GSM8K](https://github.com/openai/grade-school-math).

## Install
In order to install SGLang and other dependencies:

    cd sglang/python
    pip install .
    pip install outlines==0.0.44

One can also install SGLang through its official repo, but it may not support our process reward model, hence could only be used for sampling.

## Finetune
Our finetuning code for policy models and reward models is based on [gpt-accelera](https://github.com/Edward-Sun/gpt-accelera)
You can check the code in the finetune directory, we also provide huggingface finetune code for policy model.
You can find the models on huggingface: [Llemma-7b](https://huggingface.co/tkitsers/Llemma-metamath-7b), 
[Llemma-34b](https://huggingface.co/tkitsers/Llemma-metamath-34b), [Llemma reward model](https://huggingface.co/tkitsers/Llemma-reward-model).


## Launch Server
You can use **tmux** to start the servers, or run them in the background by adding **&** at the end of the scripts.
Make sure to set the correct paths on your device.

    bash ./scripts/run_policy.sh
    bash ./scripts/run_reward.sh

## Sampling Baseline
    bash ./scripts/sgl_baseline.sh
    bash ./scripts/hf_scores.sh

## REBASE
Before starting the REBASE, set the hyperparameters in the YAML file. Then run:

    bash ./scripts/rebase.sh

## Evaluate
GSM8K https://huggingface.co/datasets/openai/gsm8k
MATH500 https://github.com/openai/prm800k/tree/main/prm800k/math_splits/test.jsonl

You can select various aggregation functions for the scores at each step, such as last, mean, prod, or min. Additionally, you can modify the script to select answer based on best-of-n or weighted majority voting.

    bash ./scripts/evaluate.sh

## Citation
If you find our work helpful, please consider citing us:

    @misc{wu2024inferencescalinglawsempirical,
      title={Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models}, 
      author={Yangzhen Wu and Zhiqing Sun and Shanda Li and Sean Welleck and Yiming Yang},
      year={2024},
      eprint={2408.00724},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2408.00724}, 
    }

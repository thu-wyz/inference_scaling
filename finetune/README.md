# Finetune
## gpt-accelera
Using gpt-accelera, first download and convert hf model to checkpoints:

    bash ./scripts_finetune/prepare*.sh

Then finetune the reward model or policy model:

    bash ./scripts_finetune/finetune_rm.sh
    bash ./scripts_finetune/finetune_sft.sh

Finally, convert back to hf model:

    bash ./scripts_finetune/convert.sh

## huggingface
Using huggingface implementation, edit deepspeed_config.json, then run

    bash ./hf_finetune.sh


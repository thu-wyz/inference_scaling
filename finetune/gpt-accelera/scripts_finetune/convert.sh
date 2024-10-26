set -x
set -e

MODEL_REPO=llemma_7b_rm_finetune_lr-2e-5_bsz-128_score-hard_CE

python -u convert_checkpoint_to_hf.py \
    --tp_ckpt_name "/nobackup/users/zhiqings/yangzhen/models/checkpoints/${MODEL_REPO}" \
    --pretrain_name "/nobackup/users/zhiqings/yangzhen/models/checkpoints/EleutherAI/llemma_7b" \
    --tokenizer_name "EleutherAI/llemma_7b" \
    --save_name_hf "/nobackup/users/zhiqings/yangzhen/models/${MODEL_REPO}"

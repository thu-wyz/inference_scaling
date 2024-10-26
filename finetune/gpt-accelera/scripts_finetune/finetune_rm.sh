set -e
set -x

export DATA_DIR=/yangzhen/models
export MODEL_REPO=EleutherAI/llemma_7b

DEVICES=$1
NUM_DEVICE=$2
LR=$3
SCORE_TYPE=$4
BS_PER_DEVICE=$5


DATASET="reweard_dataset_path"

BSZ=128

CUDA_VISIBLE_DEVICES=$DEVICES torchrun --standalone --nproc_per_node=$NUM_DEVICE \
    finetune_rm.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --source_max_len 1024 \
    --target_max_len 1024 \
    --total_max_len 1024 \
    --per_device_train_batch_size $BS_PER_DEVICE \
    --micro_train_batch_size $BS_PER_DEVICE \
    --learning_rate $LR \
    --lr_eta_min 2e-7 \
    --num_train_epochs 2 \
    --dataset $DATASET \
    --dataset_format "reward_label" \
    --save_strategy epoch \
    --save_total_limit 1 \
    --param_dtype fp32 \
    --optim_dtype fp32 \
    --train_on_every_token \
    --tensor_parallel_size 1 \
    --save_only_model True \
    --save_dir $DATA_DIR/checkpoints/llemma_7b_rm_finetune_lr-${LR}_bsz-${BSZ}_score-${SCORE_TYPE}_CE \
    --resume_from_checkpoint
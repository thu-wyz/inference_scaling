set -e
set -x

export DATA_DIR="data_dir"
export STORE_DIR="store_path"
export MODEL_REPO="model_repo: like EleutherAI/llemma_34b"
export OMP_NUM_THREADS=8

LR=5e-6

torchrun --standalone --nproc_per_node=4 \
    finetune.py \
    --do_train \
    --checkpoint_path $DATA_DIR/checkpoints/$MODEL_REPO/model.pth \
    --source_max_len 768 \
    --target_max_len 768 \
    --total_max_len 768 \
    --per_device_train_batch_size 32 \
    --micro_train_batch_size 16 \
    --learning_rate ${LR} \
    --lr_eta_min 5e-8 \
    --num_train_epochs 1 \
    --dataset "dataset/metamath/metamath_shepherd.json" \
    --dataset_format "metamath-shepherd" \
    --add_eos_to_target \
    --save_strategy steps \
    --save_steps 200\
    --save_total_limit 1 \
    --save_dir $STORE_DIR/checkpoints/llemma-34b_metamath_shepherd \
    --tensor_parallel_size 4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.02 \
    --param_dtype bf16 \
    --optim_dtype fp32 \
    --optimizer_cpu_offload True \
    --tensor_parallel_size 4 \
    --sequence_parallel \
    --resume_from_checkpoint\

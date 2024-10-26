BASE_DIR=.
TRAIN_FILE=".jsonl"
MODEL="model_path"
CONFIG=${BASE_DIR}/deepspeed_config.json

OUTDIR="save_path"

deepspeed --include localhost:0,1,2,3,4,5,6,7  ${BASE_DIR}/hf_train.py \
    --deepspeed ${CONFIG} \
    --model_name_or_path ${MODEL} \
    --train_data_path ${TRAIN_FILE} \
    --fp16 \
    --learning_rate 8e-5\
    --output_dir ${OUTDIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --logging_dir "$OUTDIR" 

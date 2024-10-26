set -e
set -x

export DATA_DIR=/usr1/data/yangzhew/dataset/prm800k/prm800k/math_splits/test.jsonl
export OUT_DIR=./exp_results/sgl_baseline

MAX_TOKENS=1024

m_type="llemma_7b"

echo "Temperature:"
read -r TEMPERATURE

echo "SAMPLING_NUM:"
read -r SAMPLING_NUM

echo "NUM_THREADS:"
read -r NUM_THREADS

echo "HOST:"
read -r HOST

python3 sgl_baseline.py --input_path $DATA_DIR \
        --output_path $OUT_DIR/$m_type-T_$TEMPERATURE-samp_num_$SAMPLING_NUM.json \
        --sampling_num $SAMPLING_NUM \
        --max_tokens $MAX_TOKENS \
        --num_threads $NUM_THREADS \
        --temperature $TEMPERATURE \
        --policy_host http://localhost:$HOST \
        --model_type $m_type

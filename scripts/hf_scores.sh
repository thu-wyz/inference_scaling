set -e
set -x
INPUT=.json
OUTPUT=.json
MODEL=reward_model_path
python3 hf_score.py --input_path $INPUT\
    --output_path $OUTPUT\
    --batch_size 4\
    --dev 0\
    --model_path $MODEL\
    --shepherd True

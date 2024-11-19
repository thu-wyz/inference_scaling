set -e 
set -x

export path="results path"
 

python3 ./math_evaluate.py   --path $path/answers.json \
    --agg_func majority_vote \
    --model_type llemma \
    --output_path $path/results.txt

python3 ./math_evaluate.py   --path $path/answers.json \
    --agg_func majority_vote \
    --output_path $path/results.txt \
    --model_type llemma \
    --weighted True \
    --weight_agg last

python3 ./math_evaluate.py   --path $path/answers.json \
    --agg_func last \
    --model_type llemma \
    --output_path $path/results.txt


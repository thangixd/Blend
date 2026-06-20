if [ "$#" -ne 2 ]; then
    echo "Usage: ./gen_questions.sh <dataset> <number_of_questions>"
    exit
fi
project_dir="$(dirname "$PWD")"
work_dir="$(dirname "${project_dir}")"
export PYTHONPATH=${project_dir}
python generate.py \
    --work_dir ${work_dir} \
    --dataset $1 \
    --total $2
    

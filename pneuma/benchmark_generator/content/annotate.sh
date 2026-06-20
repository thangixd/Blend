if [ "$#" -ne 1 ]; then
    echo "Usage: ./annotate.sh <dataset>"
    exit
fi
work_dir="$(dirname "$PWD")"
export PYTHONPATH=${work_dir}
python annotate_tables.py \
    --work_dir ${work_dir} \
    --dataset $1 \
    

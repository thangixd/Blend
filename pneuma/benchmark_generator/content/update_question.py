import argparse
import json

def read_data(dataset):
    input_file = f'output/{dataset}/{dataset}_questions_annotated.jsonl'
    with open(input_file) as f:
        for line in f:
            item = json.loads(line)
            yield item

def main():
    args = get_args()
    out_file = f'output/{args.dataset}/questions.jsonl'
    del_keys = ['answer_tables', 'filter_by_title', 'filter_by_cell']
    with open(out_file, 'w') as f_o:
        for item in read_data(args.dataset):
            for key in del_keys:
                if key in item:
                    del item[key]
            f_o.write(json.dumps(item) + '\n')
    print(f'output to {out_file}')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__  == '__main__':
    main()

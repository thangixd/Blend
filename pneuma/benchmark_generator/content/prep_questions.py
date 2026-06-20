import os
import argparse
import json
import glob
import pandas as pd
from tqdm import tqdm
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    out_dir = f'./output/{args.dataset}/question_parts'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_file = os.path.join(out_dir, f'{args.dataset}_questions.csv')
    out_col_names = ['id', 'table_id', 'table_title', 
                     'q_no', 'question', 
                     'answer_row', 'answer_row(grid)', 
                     'answer_col', 'answer_col(grid)', 
                     'where cols', 'where_cols(grid)']
    out_data = []
    query_file_pattern = f'./output/{args.dataset}/query_*.jsonl'
    query_file_lst = glob.glob(query_file_pattern)
    random.shuffle(query_file_lst)
    for query_file in tqdm(query_file_lst):
        q_no = 0
        with open(query_file) as f:
            for line in f:
                query_info = json.loads(line)
                meta_info = query_info['meta']
                question = query_info['question']
                if question[0] == '|':
                    question = question[1:].strip()
                q_no += 1
                out_item = [
                        query_info['id'],
                        meta_info['table_id'],
                        meta_info['title'],
                        q_no,
                        question,
                        meta_info['row'],
                        meta_info['row'] + 2,
                        meta_info['sel_col'],
                        numer_to_letter(meta_info['sel_col'] + 1),
                        ' , '.join([str(a) for a in meta_info['where_cols']]),
                        ' , '.join([numer_to_letter(a+1) for a in meta_info['where_cols']])
                ] 
                out_data.append(out_item)
    
    df = pd.DataFrame(data=out_data, columns=out_col_names)
    df.to_csv(out_file)

#col_no start from 1
def numer_to_letter(col_no):
    col = col_no
    col_str = ""
    while col > 0:
        a = int((col - 1) / 26)
        b = (col -1) % 26
        col_str = chr(b + 65) + col_str
        col = a
    return col_str


if __name__ == '__main__':
    main()


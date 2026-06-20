import os
import argparse
from tqdm import tqdm
import json
from llama_gen import LlamaGenerator
from chatgpt_gen import ChatgptGenerator
from question_gen import CtrlProb
import pandas as pd
import random
import glob
import transformers
import util
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--total', type=int, default=100)
    args = parser.parse_args()
    return args

def read_tables(args):
    #nyc example table : 268n-a7em.jsonl
    #chembel example table : irac classification
    file_pattern = os.path.join(args.work_dir, 'data',
                                args.dataset, 'tables', 'tables.jsonl')
    table_file_lst = glob.glob(file_pattern)
    table_dict = {}
    #tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    for table_file in tqdm(table_file_lst, disable=(len(table_file_lst)==1)):
        with open(table_file) as f:
            for line in f:
                table_data = json.loads(line)
                #if table_data['documentTitle'] != 'MLB_42':
                #    continue
                table_id = table_data['tableId']
                table_dict[table_id] = table_data
    return table_dict

def read_state(state_file):
    if not os.path.isfile(state_file):
        return None
    with open(state_file) as f:
        state = json.load(f)
    return state

def write_state(state_file, state):
    with open(state_file, 'w') as f_o:
        f_o.write(json.dumps(state))

def main():
    args = get_args()
    out_dir = f'./output/{args.dataset}'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_question_file = os.path.join(out_dir, 'questions.jsonl')
    if os.path.isfile(out_question_file):
        cnt_opt = input(f'{out_question_file} already exists. Continue(y/n)?')
        if cnt_opt != 'y':
            return
    use_llama = 0
    if use_llama:
        generator = LlamaGenerator(args.dataset, './prompt')
    else:
        generator = ChatgptGenerator(args.dataset, './prompt')

    print('Read tables')
    table_dict = read_tables(args)
    table_id_lst = list(table_dict.keys())
    NUM_MAX = args.total
    num_questions = 0
    print('Generate questions')
    pbar = tqdm(total=NUM_MAX)
    with open(out_question_file, 'w') as f_o:
        while True:
            table_id = random.sample(table_id_lst, 1)[0]
            #print('Table (' + table_id + ') (' + table_dict[table_id]['documentTitle'] + ')')
            table_data = table_dict[table_id]
            #t1 = time.time()
            question_lst = generator.generate_questions(table_data)
            #t2 = time.time()
            #print('time ', t2 - t1)
            for q_info in question_lst:
                f_o.write(json.dumps(q_info) + '\n')
            
            pbar.update(len(question_lst))
            num_questions += len(question_lst)
            if num_questions >= NUM_MAX:
                break

if __name__ == '__main__':
    main()

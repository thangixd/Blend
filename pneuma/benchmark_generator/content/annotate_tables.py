import json
import os
import argparse
import glob
import util
from tqdm import tqdm
import gpt
from openai import OpenAI
import numpy as np
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    return args

def index_table_by_schema(col_table_map, table_data, referred_col_name_set):
    table_id = table_data['tableId']
    col_data = table_data['columns']
    for col_info in col_data:
        col_text = util.norm_text(col_info['text'])
        if col_text != '':
            if col_text not in referred_col_name_set:
                continue
            if col_text not in col_table_map:
                col_table_map[col_text] = set()
            same_col_table_set = col_table_map[col_text]
            same_col_table_set.add(table_id)

def init_worker(args):
    return

def process_table(table_file, referred_col_name_set, col_table_map):
    with open(table_file) as f:
        for line in f:
            table_data = json.loads(line)
            index_table_by_schema(col_table_map, table_data, referred_col_name_set)
            
def read_questions(args):
    q_file = f'./output/{args.dataset}/questions.jsonl'
    q_item_lst = []
    with open(q_file) as f:
        for line in f:
            q_item = json.loads(line)
            q_item_lst.append(q_item)
    return q_item_lst

def get_schema_col_names(meta):
    col_name_lst = []
    sql_struct = meta['sql_struct']
    sel_col_names = get_sel_columns(sql_struct)
    col_name_lst.extend(sel_col_names)
    where_col_names = get_where_columns(sql_struct)
    col_name_lst.extend(where_col_names)
    group_by_col_name = get_group_by_column(sql_struct)
    if group_by_col_name is not None:
        col_name_lst.append(group_by_col_name)
    having_col_name = get_having_column(sql_struct)
    if having_col_name is not None:
        col_name_lst.append(having_col_name)
    order_by_col_name = get_order_by_column(sql_struct)
    if order_by_col_name is not None:
        col_name_lst.append(order_by_col_name)
    updated_col_names = list(set(col_name_lst))
    return updated_col_names

def get_schema_shared_tables(schema_col_names, col_table_map):
    col_table_set = col_table_map[schema_col_names[0]]
    N_Cols = len(schema_col_names)
    for i in range(1, N_Cols):
        other_col_table_set = col_table_map[schema_col_names[i]]
        col_table_set = col_table_set.intersection(other_col_table_set)
    return col_table_set

def update_shared_tables(q_data, col_table_map):
    for q_item in tqdm(q_data):
        meta = q_item['meta']
        schema_col_names = get_schema_col_names(meta)
        shared_table_set = get_schema_shared_tables(schema_col_names, col_table_map)
        src_table_id = meta['table_id']
        assert src_table_id in shared_table_set
        other_table_set = shared_table_set - set([src_table_id])
        q_item['answer_tables'] = [src_table_id] + list(other_table_set)

def get_sel_columns(sql_struct):
    sel_lst = sql_struct.get('select', [])
    col_name_lst = [util.norm_text(a['col_name']) for a in sel_lst if 'col_name' in a]
    return col_name_lst

def get_where_columns(sql_struct):
    where_lst = sql_struct.get('where', [])
    col_name_lst = [util.norm_text(a['col_name']) for a in where_lst]
    return col_name_lst

def get_group_by_column(sql_struct):
    group_by = sql_struct.get('group_by', None)
    if group_by is None:
        return None
    col_name = util.norm_text(group_by['col_name'])
    return col_name

def get_having_column(sql_struct):
    having = sql_struct.get('having', None)
    if having is None:
        return None
    col_name = util.norm_text(having['col_name'])
    return col_name

def get_order_by_column(sql_struct):
    order_by = sql_struct.get('order_by', None)
    if order_by is None:
        return None
    col_name = util.norm_text(order_by['col_name'])
    return col_name

def get_referred_col_names(q_data):
    col_name_lst = []
    for q_item in q_data:
        item_col_names = get_schema_col_names(q_item['meta'])
        col_name_lst.extend(item_col_names)
    col_name_set = set(col_name_lst)
    return col_name_set

def load_prompt(promt_name):
    prompt_file = './prompt/' + promt_name + '.pmt'
    with open(prompt_file) as f:
        prompt = f.read()
    return prompt

def get_batch_tables(table_lst, batch_size):
    N = len(table_lst)
    for i in range(0, N, batch_size):
        batch_tables = table_lst[i:(i+batch_size)]
        yield batch_tables

def filter_by_title(q_item, refer_table_dict):
    meta = q_item['meta']
    if len(meta['title_ner']) == 0:
        return
    sample_entity = meta['sample_entity']
    src_table_id = meta['table_id']
    shared_table_lst = q_item['answer_tables']
    table_lst_to_check = [a for a in shared_table_lst if a != src_table_id]
    if len(table_lst_to_check) == 0:
        return
    batch_size = 20
    seq_no = 0
    prompt_template = load_prompt('entity_in_text')
    answer_dict = {}
    for batch_tables in get_batch_tables(table_lst_to_check, batch_size):
        check_entity_in_other_title(sample_entity, prompt_template, batch_tables,
                                    refer_table_dict, answer_dict)
    good_table_lst = []
    for table_id in answer_dict:
        try_lst = answer_dict[table_id]
        if np.mean(try_lst) > 0.5:
            good_table_lst.append(table_id)
    
    if len(good_table_lst) != len(table_lst_to_check):
        q_item['filter_by_title'] = len(table_lst_to_check) - len(good_table_lst)
    q_item['answer_tables'] = [src_table_id] + good_table_lst
    
def check_entity_in_other_title(sample_entity, 
                                prompt_template, 
                                batch_tables, refer_table_dict, answer_dict):
    prompt = prompt_template
    tag = 'Answer:'
    seq_no = 0
    for table_id in batch_tables:
        seq_no += 1
        table_data = refer_table_dict[table_id]
        other_title = table_data['documentTitle']
        prompt += f'\n{seq_no}. Is ` ' + sample_entity + ' ` mentioned in the sentence ` ' + other_title + ' ` ? Extract the mentioned entity from the sentence or report NO.'
    
    llm_messages[-1]['content'] = prompt
    response = gpt.chat_complete(llm_client, llm_messages, 'check_entity_in_other_title')

    out_text_lst = response.split('\n')
    for line in out_text_lst:
        awr_pos = line.find(tag)
        if awr_pos < 0:
            continue
        q_no_pos = line.find('.', 0, awr_pos)
        if q_no_pos < 0:
            continue
        sep = ';'
        sep_pos = line.find(';', awr_pos)
        answer = util.norm_text(line[awr_pos+len(tag):sep_pos])
        assert answer in ['yes', 'no']
        q_no = int(line[:q_no_pos])
        table_id = batch_tables[q_no - 1]
        if table_id not in answer_dict:
            answer_dict[table_id] = []
        try_lst = answer_dict[table_id]
        try_flag = (1 if answer == 'yes' else 0)
        try_lst.append(try_flag)

def filter_by_cell(q_item, refer_table_dict):
    meta = q_item['meta']
    cond_lst = meta['sql_struct'].get('where', [])
    eq_cond_offsets = [cond_offset for cond_offset, cond_item 
                       in enumerate(cond_lst) if cond_item['op'] == '=']
    if len(eq_cond_offsets) == 0:
        return
    shared_table_lst = q_item['answer_tables']
    src_table_id = meta['table_id']
    table_lst_to_check = [a for a in shared_table_lst if a != src_table_id]
    src_table_row_set = set()
    other_answer_table_row_dict = {}
    for cond_offset in eq_cond_offsets:
        cond_item = cond_lst[cond_offset]
        other_answer_table_row_dict[cond_offset] = set()
        cond_row = cond_item.get('row', -1)
        table_row = f'{src_table_id}@{cond_row}'
        src_table_row_set.add(table_row)
        col_name = util.norm_text(cond_item['col_name'])
        col_value = util.norm_text(cond_item['val'])
        for table_id in table_lst_to_check:
            other_table_data = refer_table_dict[table_id]
            answer_row_set = check_cell_value(other_table_data, col_name, col_value)
            if len(answer_row_set) > 0:    
                answer_table_row_lst = [f'{table_id}@{a}' for a in answer_row_set]
                other_answer_table_row_dict[cond_offset].update(answer_table_row_lst)

    answer_table_row_dict = merge_table_row(src_table_row_set, other_answer_table_row_dict)
    label_table_lst = list(answer_table_row_dict.keys())
    num_removed_tables = len(shared_table_lst) - len(label_table_lst)
    if num_removed_tables > 0:
        q_item['filter_by_cell'] = num_removed_tables
    q_item['answer_rows'] = answer_table_row_dict
    q_item['answer_tables'] = label_table_lst
    
def merge_table_row(src_answer_set, other_table_row_dict):
    other_set_lst = list(other_table_row_dict.values())
    other_answer_set = set()
    if len(other_table_row_dict) > 0:
        other_answer_set = other_set_lst[0]
        for i in range(1, len(other_set_lst)):
            next_other_set = other_set_lst[i]
            other_answer_set = other_answer_set.intersection(next_other_set)

    table_row_dict = {}
    merge_answer_set = src_answer_set.union(other_answer_set)
    for table_row in merge_answer_set:
        pos = table_row.rindex('@')
        table = table_row[:pos]
        row = int(table_row[pos+1:])
        if table not in table_row_dict:
            table_row_dict[table] = []
        table_row_dict[table].append(row)
    return table_row_dict

def check_cell_value(other_table_data, col_name, col_value):
    other_col_data = other_table_data['columns']
    other_row_data = other_table_data['rows']
    col_lst = []
    for col in range(len(other_col_data)):
        if util.norm_text(other_col_data[col]['text']) == col_name:
            col_lst.append(col)

    answer_row_set = set()
    for col in col_lst:
       for row, row_item in enumerate(other_row_data):
            cell_info = row_item['cells'][col]
            other_value = util.norm_text(cell_info['text'])
            matched = False
            if other_value == col_value:
                matched = True
            else:
                if util.is_float(other_value):
                    other_value = str(float(other_value))
                    if other_value == col_value:
                        matched = True
            if matched:
                answer_row_set.add(row)
    return answer_row_set

def get_out_file(args):
    out_file_name = args.dataset + '_questions_annotated.jsonl'
    out_file = os.path.join('output', args.dataset, out_file_name)
    return out_file

def filter_tables(args, q_data, refer_table_dict):
    for q_item in tqdm(q_data):
        if q_item['meta']['sql_struct']['options']['use_title']:
            filter_by_title(q_item, refer_table_dict)
        filter_by_cell(q_item, refer_table_dict)

    out_file = get_out_file(args)
    with open(out_file, 'w') as f_o:
        for q_item in q_data:
            f_o.write(json.dumps(q_item) + '\n')

def read_referred_tables(args, q_data):
    refer_table_dict = {}
    table_set = set()
    for q_item in q_data:
        table_set.update(set(q_item['answer_tables']))
    
    file_pattern = os.path.join(args.work_dir, 'data', args.dataset, 'tables/tables.jsonl')
    table_file_lst = glob.glob(file_pattern)
    for table_file in table_file_lst:
        with open(table_file) as f:
            for line in tqdm(f):
                table_data = json.loads(line)
                table_id = table_data['tableId']
                refer_table_dict[table_id] = table_data
    return refer_table_dict

def init_llm(args):
    api_key = os.getenv('OPENAI_API_KEY', None)
    if api_key is None:
        raise ValueError('Need to set environment variable OPENAI_API_KEY')
    global llm_client
    llm_client = OpenAI(api_key=api_key)

    global llm_messages
    llm_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": None},
    ]
    init_log_setting(args)

def init_log_setting(args):
    now_t = str(datetime.now())
    log_name = 'log_annotate_' + '_'.join(now_t.split()) + '.txt'
    gpt_log_dir = os.path.join('./prompt/log/' + args.dataset, 'chatgpt')
    if not os.path.isdir(gpt_log_dir):
        os.makedirs(gpt_log_dir)
    log_file = os.path.join(gpt_log_dir, log_name)
    f_log = open(log_file, 'w')
    gpt.set_logger(f_log)

def main():
    args = get_args()
    out_file = get_out_file(args)
    if os.path.isfile(out_file):
        print(f'{out_file} already exists')
        return
    init_llm(args)
    print('Read questions')
    q_data = read_questions(args)
    print('Search schema-shared tables')
    referred_col_name_set = get_referred_col_names(q_data)
    file_pattern = os.path.join(args.work_dir, 'data', args.dataset, 'tables/tables.jsonl')
    table_file_lst = glob.glob(file_pattern)
    col_table_map = {}
    for table_file in tqdm(table_file_lst, disable=(len(table_file_lst)==1)):
        process_table(table_file, referred_col_name_set, col_table_map)
    
    update_shared_tables(q_data, col_table_map)

    print('Read referred tables')
    refer_table_dict = read_referred_tables(args, q_data)

    print('Filter answer tables')
    filter_tables(args, q_data, refer_table_dict)

if __name__ == '__main__':
    main()
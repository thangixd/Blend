import json
import csv
import argparse
import os
from tqdm import tqdm
from multiprocessing import Pool as ProcessPool
import uuid
import glob
import random
import sys

MAX_COL_NAME_SIZE = 64
MAX_CELL_SIZE = 128

csv.field_size_limit(sys.maxsize)

def get_out_file(args):
    data_dir = os.path.join(args.work_dir, 'data/%s/tables' % args.dataset)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    data_file = os.path.join(data_dir, 'tables.jsonl')
    err_file = os.path.join(data_dir, 'err.txt')
    return data_file, err_file

def read_meta(meta_file):
    if not os.path.exists(meta_file):
        return None
    with open(meta_file) as f:
        meta_data = json.load(f)
    return meta_data

def read_table(arg_info):
    csv_file = arg_info['data_file']
    meta_file = arg_info['meta_file']
    file_name = os.path.basename(os.path.splitext(csv_file)[0])
    truncate_flag = arg_info['truncate']
    table_title = '' 
    table_id = ''
    sep_tag = '_SEP_'
    table_meta = read_meta(meta_file)
    if table_meta is not None:
        table_id = table_meta['table_id']
        table_title = table_meta['title']
    else:
        if sep_tag in file_name:
            tag_info = file_name.split(sep_tag)
            table_title = tag_info[0]
            table_id = tag_info[1]

    if table_title == '':
        if arg_info['file_name_title']:
            table_title = file_name
    if table_id == '':
        table_id = file_name + '_' + str(uuid.uuid4())
    table = {
        'columns':None,
        'rows':[],
        'tableId':table_id,
        'documentTitle':table_title,
    }
    col_name_lst = None
    unk_col_set = set()
    err_msg_lst = []
    with open(csv_file) as f:
        reader = csv.reader(f, delimiter=',')
        row_data = table['rows']
        for row, item in enumerate(reader):
            if row == 0:
                if truncate_flag:
                    col_name_lst = [process_col_name(a, col_offset, unk_col_set, err_msg_lst)
                                    for col_offset , a in enumerate(item)]
                    correct_col_repeat(col_name_lst, csv_file, err_msg_lst)
                else:
                    col_name_lst = item
            else:
                if len(item) != len(col_name_lst):
                    err_text = f'row {row} in file {csv_file} contains less columns'
                    raise ValueError(err)
                if truncate_flag:
                    cells = [{'text':process_cell(a, col_offset, row, err_msg_lst)}
                            for col_offset, a in enumerate(item)]
                else:
                    cells = [{'text':a} for a in item]
                cell_info = {'cells':cells}
                row_data.append(cell_info)
    
    if table_meta is not None:
        meta_col_names = table_meta.get('col_names', None)
        if meta_col_names is not None:
            assert len(col_name_lst) == len(meta_col_names)
            col_name_lst = meta_col_names
    
    table['columns'] = [{'text':col_name} for col_name in col_name_lst]
    col_data = table['columns']
    for col in unk_col_set:
        col_data[col]['unk'] = True
    table['_err_'] = {'msg':err_msg_lst, 'file_name':csv_file}
    return table

def auto_fix_col_names(max_suffix_size, updated_name_dict, 
                      repeat_col_lst, col_name_lst, 
                      err_lst):
    suffix = 0
    for col in repeat_col_lst:
        suffix += 1
        if suffix == 1:
            new_col_name = col_name_lst[col]
        else:
            max_try = 0
            while True:
                max_try += 1
                suffix_text = '_' + str(suffix)
                if len(col_name_lst[col] + suffix_text) < MAX_COL_NAME_SIZE:
                    new_col_name = col_name_lst[col] + suffix_text
                else:
                    new_col_name = col_name_lst[col][:-max_suffix_size] + suffix_text
                if (new_col_name.lower() not in updated_name_dict) or (max_try >= 6):
                    break
                suffix += 1

        new_key = new_col_name.lower()
        if new_key in updated_name_dict:
            raise ValueError(f'can not fix column {col}')
        else:
            updated_name_dict[new_key] = True
            if suffix > 1:
                err_msg = f'column {col} is renamed because of repeat'
                err_lst.append(err_msg)
                col_name_lst[col] = new_col_name

def correct_col_repeat(col_name_lst, csv_file, err_lst):
    name_dict = {}
    for col, col_name in enumerate(col_name_lst):
        key = col_name.lower()
        if key not in name_dict:
            name_dict[key] = []
        same_col_lst = name_dict[key]
        same_col_lst.append(col)
    updated_name_dict = {}
    repeat_exists = False
    max_atuto_fix_repeat_size = 100
    max_suffix_size = 3 # e.g. _99
    for key in name_dict:
        repeat_col_lst = name_dict[key]
        repeat_size = len(repeat_col_lst)
        if repeat_size <= 1:
            updated_name_dict[key] = True
            continue
        repeat_exists = True
        can_fix = True
        if repeat_size < max_atuto_fix_repeat_size:
            auto_fix_col_names(max_suffix_size, updated_name_dict, 
                               repeat_col_lst, col_name_lst,
                               err_lst)
        else:
            can_fix = False
        if not can_fix:
            repeat_name = col_name_lst[repeat_col_lst[0]]
            err = '\n' + ('-' * 60)
            err += '\nNeed to correct column names'
            err += f'\nThere are {repeat_size} columns with the same name (up to size {MAX_COL_NAME_SIZE}) \n "{repeat_name}" \n in {csv_file}'
            err += '\n' + ('-' * 60)
            raise ValueError(err)
    if repeat_exists:
        assert len(updated_name_dict) == len(col_name_lst)

def process_col_name(text, col, unk_col_set, err_lst):
    updated_text = text.strip()
    if updated_text == '' or updated_text == '`':
        unk_col_set.add(col)
        updated_text = 'unk_' + str(len(unk_col_set))

    if updated_text.find(',') > -1:
        err = 'comma , in column {col} name is replaced with period . becasue of My SQL'
        err_lst.append(err)
        updated_text = updated_text.replace(',', '.')
    
    if updated_text.find('(') > -1:
        err = 'char ( in column {col} name is replaced with space becasue of My SQL'
        err_lst.append(err)
        updated_text = updated_text.replace('(', ' ')

    if updated_text.find(')') > -1:
        err = 'char ) in column {col} name is replaced with space becasue of My SQL'
        err_lst.append(err)
        updated_text = updated_text.replace(')', ' ')

    updated_text = ' '.join(updated_text.split())
    text_size = len(updated_text)
    if text_size > MAX_COL_NAME_SIZE:
        err = f'column {col} name size {text_size} > {MAX_COL_NAME_SIZE}'
        err_lst.append(err)
        updated_text = truncate_text(updated_text, MAX_COL_NAME_SIZE)
    return updated_text

def truncate_text(text, max_size):
    input_word_lst = text.split()
    out_text = ''
    for word in input_word_lst:
        if out_text == '':
            update_out_text = word
        else:
            update_out_text = out_text + ' ' + word
        if len(update_out_text) <= max_size:
            out_text = update_out_text
        else:
            break
    if len(out_text) == 0:
        out_text = text[:max_size]
    return out_text

def process_cell(text, col, row, err_lst):
    updated_text = text.strip()
    text_size = len(updated_text)
    if text_size > MAX_CELL_SIZE:
        err = f'cell at row {row} col {col} size {text_size} > {MAX_CELL_SIZE}'
        err_lst.append(err)
        updated_text = truncate_text(updated_text, MAX_CELL_SIZE)
    return updated_text

def show_args(args):
    arg_str_lst = []
    for arg in vars(args):
        arg_val = getattr(args, arg)
        if type(arg_val) == str:
            arg_str = f"{arg}='{arg_val}'"
        else:
            arg_str = f"{arg}={arg_val}"
        arg_str_lst.append(arg_str)
    str_info = 'Args(' + ', '.join(arg_str_lst) + ')'
    print(str_info)
    
def main(args):
    out_file, err_file = get_out_file(args)
    if os.path.exists(out_file):
        msg_text = '%s already exists' % out_file
        print(msg_text)
        msg_info = {
            'state':False,
            'msg':msg_text
        }
        return msg_info
    show_args(args)
    f_o = open(out_file, 'w')
    f_err = open(err_file, 'w')
    dataset_dir = os.path.join(args.work_dir, 'data', args.dataset)
    csv_file_pattern = os.path.join(dataset_dir, '**', '*.csv')
    csv_file_lst = glob.glob(csv_file_pattern, recursive=True)
   
    num_wokers = min(os.cpu_count(), 10) 
    work_pool = ProcessPool(num_wokers)
    arg_info_lst = []
    
    for csv_file in csv_file_lst:
        meta_file = os.path.splitext(csv_file)[0] + '.meta.json'
        args_info = {
            'data_file':csv_file,
            'meta_file':meta_file,
            'file_name_title':args.file_name_title,
            'truncate':args.truncate
        }
        arg_info_lst.append(args_info)

    err_exists = False
    multi_process = True
    if multi_process:    
        for table in tqdm(work_pool.imap_unordered(read_table, arg_info_lst), total=len(arg_info_lst)):
            if table is None:
                continue
            if len(table['_err_']['msg']) > 0:
                err_exists = True
            output_table(table, args, f_o, f_err)
    else:
        for arg_info in tqdm(arg_info_lst):
            table = read_table(arg_info)
            if table is None:
                continue
            if len(table['_err_']['msg']) > 0:
                err_exists = True
            output_table(table, args, f_o, f_err)
     
    f_o.close()
    f_err.close()
    if err_exists:
        print(f'Check warnings in {err_file}')

    msg_info = {
        'state':True,
    }
    return msg_info

def output_table(table, args, f_o, f_err):
    err_info = table['_err_']
    err_msg_lst = err_info['msg']
    file_name = err_info['file_name']
    if len(err_msg_lst) > 0:
        err_text = f'error in {file_name}\n' 
        err_text += '\n'.join(err_msg_lst) + '\n'
        err_text += '-' * 100 + '\n'
        f_err.write(err_text)
    del table['_err_']
    f_o.write(json.dumps(table) + '\n')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--file_name_title', type=int, default=1)
    parser.add_argument('--truncate', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)
    
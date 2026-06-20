import json
import csv
import argparse
import os
from tqdm import tqdm
import random
import glob

def read_table_filter(data_file):
    table_lst = []
    with open(data_file) as f:
        for line in f:
            table_id = line.strip()
            table_lst.append(table_id)
    table_set = set(table_lst)
    return list(table_set)

def read_tables(args):
    table_file_pattern = os.path.join('../../data', args.dataset, 'tables/tables.jsonl') 
    table_file_lst = glob.glob(table_file_pattern)
    for table_file in table_file_lst:
        with open(table_file) as f:
            for line in f:
                table_data = json.loads(line)
                yield table_data
    
def main():
    args = get_args()
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    if len(os.listdir(args.output_dir)) > 0:
        print(f'{args.output_dir} is not empty')
        return

    for table in tqdm(read_tables(args)):
        table_id = table['tableId']
        table_title = table['documentTitle'].strip()
        caption = table_title.replace('/', ' ')
        if len(caption) > 0:
            if caption[0] == '.':
                caption = '_' + caption[1:]

        file_name = f'{caption}_SEP_{table_id}.csv'
        out_file = os.path.join(args.output_dir, file_name) 

        with open(out_file, 'w') as f_o:
            columns = table['columns']
            writer = csv.writer(f_o)
            col_names = [col_info['text'] for col_info in columns]
            writer.writerow(col_names)
            row_data = table['rows']
            for row_item in row_data:
                cells = row_item['cells']
                cell_values = [a['text'] for a in cells] 
                writer.writerow(cell_values)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()



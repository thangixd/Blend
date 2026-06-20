import json
from tqdm import tqdm
import pandas as pd
from table_ingestion import util

def read_table_columns():
    col_dict = {}
    table_dict = {}
    with open('../../data/nyc_open_1000/tables/tables.jsonl') as f:
        for line in tqdm(f):
            table_data = json.loads(line)
            table_id = table_data['tableId']
            table_dict[table_id] = table_data
            col_data = table_data['columns'][:6]
            title = table_data['documentTitle']
            col_names = [a['text'] for a in col_data]
            for offset, col_name in enumerate(col_names):
                if ('year' in col_name) or ('date' in col_name) or ('time' in col_name):
                    continue
                if ('number' in col_name) or ('score' in col_name) or ('#' in col_name):
                    continue

                key = col_name.strip().lower()
                if key not in col_dict:
                    col_dict[key] = []
                item_lst = col_dict[key]
                item = {'title':title, 'col_name':col_name, 'col_offset':offset, 'table_data': table_data}
                item_lst.append(item)
    return col_dict, table_dict

def can_join(col_key, item_lst):
    text_dict = {}
    for item in item_lst:
        table_data = item['table_data']
        col = item['col_offset']
        row_data = table_data['rows']
        table_id = table_data['tableId']
        for row_item in row_data:
            text = row_item['cells'][col]['text'].strip()
            if text == '':
                continue
            if util.is_float(text):
                continue
            key = text.lower()
            if key not in text_dict:
                text_dict[key] = set()
            table_set = text_dict[key]
            table_set.add(table_id)
    join_table_lst = []
    for text_key in text_dict:
        table_lst = list(text_dict[text_key])
        if len(table_lst) > 1:
            join_info = {'col_key':col_key, 'text_key':text_key, 'tables':table_lst} 
            join_table_lst.append(join_info)
    return join_table_lst

def main():
    out_data = []
    out_col_names = ['column name', 'cell text', 'table id', 'table caption']
    col_dict, table_dict = read_table_columns()
    for col_key in tqdm(col_dict):
        item_lst = col_dict[col_key]
        join_table_lst = can_join(col_key, item_lst)
        for join_info in join_table_lst[:10]:
            text_key = join_info['text_key']
            table_lst = join_info['tables']
            for table in table_lst:
                caption = table_dict[table]['documentTitle']
                out_item = [col_key, text_key, table, caption]
                out_data.append(out_item)

    df = pd.DataFrame(out_data, columns=out_col_names)
    df.to_csv('./output/nyc_open_1000_join_table.csv')

if __name__ == '__main__':
    main()

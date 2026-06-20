import json
import os
import sql_parser
import re
import sqlglot
from tqdm import tqdm
from annotate_tables import get_schema_col_names
import util

def write_jsonl(data: list[dict[str,str]], out_dir: str):
    f_o_sql = open(out_dir + '/bird_sql.jsonl', 'w')
    with open(out_dir + '/bird_questions.jsonl', "w", encoding="utf-8") as file:
        for offset, item in enumerate(data):
            file.write(json.dumps(item))
            file.write("\n")
            f_o_sql.write(str(offset + 1) + '.  ' + item['sql'] + '\n\n\n')
    f_o_sql.close()

def check_refer_cols(refer_col_names, col_table_map):
    for refer_col in refer_col_names:
        if refer_col not in col_table_map:
            return False
    return True

def convert_to_jsonl(data: list, out_dir: str):
    res = []
    f_err = open(out_dir + '/bird_sql_err.txt', 'w')
    table_dict = index_tables()
    datum_idx = 0
    for datum in data:
        db_id = datum["db_id"]
        sql = datum["SQL"]
        meta = get_sql_meta(sql, db_id, f_err)
        if meta is None:
            continue
        
        refer_col_names = get_schema_col_names(meta)
        table_id = meta['table_id']
        if table_id not in table_dict:
            continue
        col_dict = table_dict[table_id]
        if not check_refer_cols(refer_col_names, col_dict):
            continue

        datum_idx += 1
        res.append({
            "id": datum_idx,
            "db_id": db_id,
            "question": datum["question"],
            "sql":sql,
            "meta":meta
        })
    f_err.close()
    write_jsonl(res, out_dir)

def get_table_id(db_id, table_name):
    db_table_sep = '-#-'
    table_id = (db_id + db_table_sep + table_name).lower()
    return table_id

def get_sql_meta(sql, db_id, f_err):
    sql = sql.replace("`", "'") # sqlglot does not work with ` (ord=96)
    stmt = sql_parser.parse_sql(sql)
    table_name = stmt.find(sqlglot.expressions.Table).name
    err_text = ''
    
    select_struct = None
    if 'SELECT ' in sql:
        select_struct = sql_parser.get_select(stmt)
        if not select_struct:
            err_text += ' SELECT err'
    
    where_struct = None
    if 'WHERE ' in sql:
        where_struct = sql_parser.get_where(stmt)
        if not where_struct:
            err_text += ' WHERE err'

    group_struct = None
    if 'GROUP ' in sql:
        group_struct = sql_parser.get_group_by(stmt)
        if not group_struct:
            err_text += ' GROUP err'
    
    having_struct = None
    if 'HAVING ' in sql:
        having_struct = sql_parser.get_having(stmt)
        if not having_struct:
            err_text += ' HAVING err'
    
    order_struct = None
    if 'ORDER ' in sql:
        order_struct = sql_parser.get_order_by(stmt)
        if not order_struct:
            err_text += ' ORDER err'

    limit_struct = None
    if 'LIMIT ' in sql:
        limit_struct = sql_parser.get_limit(stmt)
        if not limit_struct:
            err_text += ' LIMIT err'

    if err_text != '':
        out_sql_err = sql + '\n' + err_text
        out_sql_err += '\n\n\n'
        f_err.write(out_sql_err)
        return None
    else:
        sql_meta = {
            'table_id':get_table_id(db_id, table_name),
            'title':table_name,
            'sql_struct':{
                'options':{
                    'use_title':0
                }
            }
        }
        sql_struct = sql_meta['sql_struct']
        if select_struct:
            sql_struct['select'] = select_struct
        if where_struct:
            sql_struct['where'] = where_struct
        if group_struct:
            sql_struct['group_by'] = group_struct
        if order_struct:
            sql_struct['order_by'] = order_struct
        if limit_struct:
            sql_struct['limit'] = limit_struct
        return sql_meta

def find_one(text, sub_str):
    pos_1 = text.find(sub_str)
    if pos_1 >= 0:
        pos_2 = text.find(sub_str, pos_1 + len(sub_str))
        if pos_2 >= 0:
            return -1
        else:
            return pos_1
    return -1

def check_simple(sql):
    if ' JOIN ' in sql:
        return False
    from_text = ' FROM '
    from_pos = find_one(sql, from_text)
    if from_pos < 0:
        return False
    pos_2 = find_one(sql, ' WHERE ')
    if pos_2 < 0:
        return False
    table_text = sql[from_pos + len(from_text):pos_2]
    if len(table_text.split(',')) > 1:
        print(table_text)
    return True

def index_table_by_col(table_data):
    col_dict = {}
    col_data = table_data['columns']
    for col_info in col_data:
        col_text = util.norm_text(col_info['text'])
        if col_text != '':
            if col_text not in col_dict:
                col_dict[col_text] = True
    return col_dict
            
def index_tables():
    table_file = '/home/cc/code/Pneuma/benchmark_generator/data/bird/tables/tables.jsonl'
    table_dict = {}
    with open(table_file) as f:
        for line in tqdm(f):
            table_data = json.loads(line)
            table_id = table_data['tableId']
            col_dict = index_table_by_col(table_data)
            table_dict[table_id] = col_dict
    return table_dict

def main():
    out_dir = 'questions'
    if os.path.isdir(out_dir):
        print(f'{out_dir} already exists')
        return
    os.mkdir(out_dir)
    with open("/home/cc/text2sql_bird/dev_20240627/dev.json") as f:
        dev = json.load(f)
    with open("/home/cc/text2sql_bird/train/train.json") as f:
        train = json.load(f)
    
    all_data = train + dev
    filtered_all = [a for a in all_data if check_simple(a['SQL'])]
    convert_to_jsonl(filtered_all, out_dir)

if __name__ == '__main__':
    main()

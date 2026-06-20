import json
import csv
from tqdm import tqdm
import uuid

def read_table_1000():
    table_data_lst = []
    data_file = '../../data/nyc_open_1000/tables/tables.jsonl'
    with open(data_file) as f:
        for line in tqdm(f):
            table_data = json.loads(line)
            table_data_lst.append(table_data)
    return table_data_lst

def read_item_one_table(item):
    table_id = item[1].strip()
    question = item[2].strip()
    grid_row_text = item[3].strip()
    grid_row_lst = grid_row_text.split(',')
    cell_row_lst = grid_row_to_cell_row(grid_row_lst) 
    return [table_id], question, [cell_row_lst]

def grid_row_to_cell_row(grid_row_lst):
    if grid_row_lst[0].lower() == 'ok':
        return []
    cell_row_lst = [int(a) - 2 for a in grid_row_lst]
    return cell_row_lst

def read_item_multi_table(item):
    table_id_text = item[1].strip()
    table_lst = table_id_text.split('\n')
    table_id_lst = [a.strip() for a in table_lst]
    question = item[2].strip()
    grid_row_text_lst = item[3].strip().split('\n')
    batch_cell_row_lst = []
    for grid_row_text in grid_row_text_lst:
        grid_row_lst = grid_row_text.split(',')
        cell_row_lst = grid_row_to_cell_row(grid_row_lst)
        batch_cell_row_lst.append(cell_row_lst)
    assert len(table_id_lst) == len(batch_cell_row_lst)
    return table_id_lst, question, batch_cell_row_lst

def read_label_file(label_file, refer_to_multi_table):
    question_label_data = []
    with open(label_file) as f:
        reader = csv.reader(f, delimiter=',')
        for row, item in enumerate(reader):
            if row == 0:
                continue
            question = item[2]
            if question.strip() == '':
                continue
            if not refer_to_multi_table:
                q_info = read_item_one_table(item)
            else:
                q_info = read_item_multi_table(item)
            question_label_data.append(q_info) 
    return question_label_data

def gen_query(q_label_data, file_name):
    with open(file_name, 'w') as f_o: 
        for q_label in q_label_data:
            table_id_lst, question, _ = q_label
            query_item = {
                "id":str(uuid.uuid4()),
                "question":question,
                "table_id_lst":table_id_lst,
                "answers":["N/A"],
                "ctxs": [{"title": "", "text": "This is a example passage."}]
            } 
            f_o.write(json.dumps(query_item) + '\n')

def gen_table_10(table_row_dict, table_1000_data, file_name):
    with open(file_name, 'w') as f_o:
        for table_data in tqdm(table_1000_data):
            table_id = table_data['tableId']
            row_data = table_data['rows']
            row_10_data = None 
            if table_id not in table_row_dict:
                row_10_data = row_data[:10]
            else:
                cell_row_set = table_row_dict[table_id]
                if len(cell_row_set) == 0:
                    row_10_data = row_data[:10]
                else:
                    row_10_offset_lst = []
                    row_offset_other = []
                    for row_offset, row_item in enumerate(row_data):
                        if row_offset in cell_row_set:
                            row_10_offset_lst.append(row_offset)
                        else:
                            row_offset_other.append(row_offset)
                    
                    N = len(row_10_offset_lst) 
                    assert(N <= 10)
                    if N < 10:
                        M = 10 - N
                        row_10_offset_lst += row_offset_other[:M] 
                
                row_10_data = [row_data[a] for a in row_10_offset_lst]
            
            assert len(row_10_data) <= 10
            table_data['rows'] = row_10_data
            f_o.write(json.dumps(table_data) + '\n')

def main():
    one_table_file = './data/nyc_open_1000_questions_labeled/nyc_open_1000_questions - question_on_one_table.csv'
    question_label_data_one_table = read_label_file(one_table_file, False)
    
    multi_table_file = './data/nyc_open_1000_questions_labeled/nyc_open_1000_questions - question_on_multiple_tables.csv'
    question_label_data_multi_table = read_label_file(multi_table_file, True)
   
    gen_query(question_label_data_one_table, './data/nyc_open_1000_questions_labeled/query_one_table.jsonl')
    gen_query(question_label_data_multi_table, './data/nyc_open_1000_questions_labeled/query_multi_table.jsonl')

    table_row_dict = {}
    label_data_lst = question_label_data_one_table + question_label_data_multi_table
    for label_data in label_data_lst:
        table_id_lst, _, batch_cell_row_lst = label_data
        for offset, table_id in enumerate(table_id_lst):
            cell_row_lst = batch_cell_row_lst[offset]
            if table_id not in table_row_dict:
                table_row_dict[table_id] = set()
            row_set = table_row_dict[table_id]
            for cell_row in cell_row_lst:
                row_set.add(cell_row)

    table_1000_data = read_table_1000()
    gen_table_10(table_row_dict, table_1000_data, './data/nyc_open_1000_questions_labeled/tables_10.jsonl')

if __name__ == '__main__':
    main()

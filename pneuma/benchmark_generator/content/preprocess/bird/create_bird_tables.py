import sqlite3
import csv
import os

from tqdm import tqdm

def create_tables(db_folder, table_id_dict, out_dir):
    table_folders = [i for i in os.listdir(db_folder)]
    for folder in tqdm(table_folders):
        if not os.path.isdir(f"{db_folder}/{folder}"):
            continue
        connection = sqlite3.connect(f"{db_folder}/{folder}/{folder}.sqlite")
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master \
                        WHERE type='table' and name != 'sqlite_sequence';")
        table_name_lst = cursor.fetchall()
        db_table_sep = '-#-'
        for table_name_tuple in table_name_lst:
            table_name = table_name_tuple[0]
            assert db_table_sep not in folder
            table_id = (f"{folder}{db_table_sep}{table_name}").lower()
            if table_id not in table_id_dict:
                table_id_dict[table_id] = True
            else:
                raise ValueError(f'{table_id} duplicate')
            with open(f"{out_dir}/{table_name}_SEP_{table_id}.csv", "w", newline="") as csv_file:
                writer = csv.writer(csv_file)

                cursor.execute(f'PRAGMA table_info("{table_name}");')
                headers = [column[1] for column in cursor.fetchall()]
                writer.writerow(headers)

                cursor.execute(f'SELECT * FROM "{table_name}"')
                writer.writerows(cursor.fetchall())
        connection.close()

def main():
    out_dir = 'tables'
    if os.path.isdir(out_dir):
        print(f'{out_dir} already exists')
        return
    os.mkdir(out_dir)
    table_id_dict = {}
    create_tables('/home/cc/text2sql_bird/train/train_databases', table_id_dict, out_dir)
    create_tables('/home/cc/text2sql_bird/dev_20240627/dev_databases', table_id_dict, out_dir)

if __name__ == '__main__':
    main()
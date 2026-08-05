import argparse
from collections import defaultdict
from glob import glob
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import duckdb
from src.utils import calculate_xash


# The batch is held in memory twice here, as tuples and as a DataFrame, so it is kept
# smaller than the 3.5M rows the Vertica script streams over a socket.
BATCH_SIZE = 1_000_000


def create_table(dbcon, table_name):
    print('Preparing index creation by creating tables in DB.')
    dbcon.execute(f'DROP TABLE IF EXISTS "{table_name}";')
    dbcon.execute(f'DROP TABLE IF EXISTS "{table_name}_tables";')

    # super_key holds a 128 character bitstring, not binary, because
    # MultiColumnOverlap.run_filter reads it back with int(value, 2).
    dbcon.execute(f"""CREATE TABLE "{table_name}" (
        tokenized VARCHAR,
        tableid INTEGER,
        colid INTEGER,
        rowid INTEGER,
        super_key VARCHAR,
        quadrant BOOLEAN
        );""")

    dbcon.execute(f"""CREATE TABLE "{table_name}_tables" (
        tableid INTEGER,
        filename VARCHAR
        );""")

    print('Preparation of index creation is finished!')


def create_indexes(dbcon, table_name):
    print('Creating indexes.')
    dbcon.execute(f'CREATE INDEX "{table_name}_to_tokenized" ON "{table_name}" (tokenized);')
    dbcon.execute(f'CREATE INDEX "{table_name}_to_tableid" ON "{table_name}" (tableid);')
    print('Creation of indexes is finished!')


def save_data_to_duckdb(dbcon, data, table_name):
    df = pd.DataFrame(data, columns=['tokenized', 'tableid', 'colid', 'rowid', 'super_key', 'quadrant'])
    # The Vertica COPY declares NULL '', so empty tokens are stored as NULL there too.
    df['tokenized'] = df['tokenized'].astype('string')
    df.loc[df['tokenized'] == '', 'tokenized'] = pd.NA
    # Booleans mixed with None arrive as an object column, which DuckDB types wrong.
    df['quadrant'] = df['quadrant'].astype('boolean')

    dbcon.register('batch_df', df)
    try:
        dbcon.execute(f'INSERT INTO "{table_name}" SELECT * FROM batch_df')
    finally:
        dbcon.unregister('batch_df')


def assign_table_ids(lake_path):
    """Maps every file in the lake to its TableId. The one place ids are assigned."""
    file_paths = sorted(glob(f'{lake_path}'))
    if not file_paths:
        raise FileNotFoundError(f'No files matched {lake_path}')

    return list(enumerate(file_paths))


def create_index(dbcon, assignment, table_name, sep=',', batch_size=BATCH_SIZE):
    create_table(dbcon, table_name)

    print(f'Inserting data into index from {len(assignment)} files.')
    dbcon.executemany(f'INSERT INTO "{table_name}_tables" VALUES (?, ?)',
                      [(table_id, Path(path).name) for table_id, path in assignment])

    data = []
    for table_counter, file_path in tqdm(list(assignment)):
        file_content_df = pd.read_csv(file_path, sep=sep, low_memory=False)
        numeric_cols = file_content_df.select_dtypes(include='number').columns
        numeric_cols = [file_content_df.columns.get_loc(col) for col in numeric_cols]

        file_content = file_content_df.values
        number_of_rows = file_content.shape[0]
        number_of_cols = file_content.shape[1]

        pbar = None
        if number_of_cols * number_of_rows > 1000000:
            pbar = tqdm(total=number_of_cols * number_of_rows, leave=False)

        superkeys = defaultdict(int)
        new_data = []
        for col_counter in range(number_of_cols):
            is_numeric_col = col_counter in numeric_cols
            if is_numeric_col:
                mean = np.nanmean(file_content[:, col_counter])
            for row_counter in range(number_of_rows):
                tokenized = str(file_content[row_counter][col_counter]).lower().replace('\\', '').replace('\'', '').replace('\"', '').replace('\t', '').replace('\n', '').replace('\r', '').strip()[0:200]
                if tokenized == 'nan' or tokenized == 'none':
                    tokenized = ''
                quadrant = file_content[row_counter][col_counter] >= mean if is_numeric_col else None
                new_data.append((tokenized, table_counter, col_counter, row_counter, quadrant))
                superkeys[row_counter] = superkeys[row_counter] | calculate_xash(str(tokenized))
                if pbar: pbar.update(1)

        superkeys_as_binary = {key: f"{superkey:0128b}" for key, superkey in superkeys.items()}
        data.extend([(x[0], x[1], x[2], x[3], superkeys_as_binary[x[3]], x[4]) for x in new_data])
        if len(data) >= batch_size:
            save_data_to_duckdb(dbcon, data, table_name)
            data = []

        if pbar: pbar.close()

    if len(data) > 0:
        save_data_to_duckdb(dbcon, data, table_name)

    print('Insertion of the index is Finished!')
    create_indexes(dbcon, table_name)


def parse_args():
    repo_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(description='Build a BLEND index in DuckDB from a CSV data lake.')
    parser.add_argument('--lake', default='adventure_works',
                        help='Lake directory inside --datalake, also used as the index table name.')
    parser.add_argument('--datalake', default=str(repo_root / 'datalake'), help='Directory holding the lakes.')
    parser.add_argument('--db', default=str(repo_root / 'blend_duckdb.db'), help='DuckDB database file to write.')
    parser.add_argument('--sep', default=',', help='CSV separator.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Index rows to buffer before writing to DuckDB.')

    return parser.parse_args()


def main():
    args = parse_args()

    lake_dir = Path(args.datalake) / args.lake
    if not lake_dir.is_dir():
        raise NotADirectoryError(f'Lake directory not found: {lake_dir}')

    # DBHandler opens DuckDB read-only, so index creation needs its own connection.
    dbcon = duckdb.connect(database=args.db, read_only=False)
    try:
        assignment = assign_table_ids(str(lake_dir / '*.csv'))
        create_index(dbcon, assignment, args.lake, sep=args.sep, batch_size=args.batch_size)
    finally:
        dbcon.close()

    print(f'\nIndex written to {args.db}. To query it, set in config/config.ini:')
    print(f'  dbms=duckdb\n  path={args.db}\n  index_table={args.lake}')


if __name__ == '__main__':
    main()

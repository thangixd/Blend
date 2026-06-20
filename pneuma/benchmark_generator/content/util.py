import re
import numpy as np
import math

class CellDataType:
    INT = 1
    BOOL = 2
    FLOAT = 3
    POLYGON = 4
    Other = 5

def is_float(text):
    strip_text = text.strip()
    if '.' not in strip_text:
        return False
    if re.match(r'^-?\d+(?:\.\d+)$', strip_text) is None:
        return False
    return True

def is_int(text):
    strip_text = text.strip()
    if strip_text == '':
        return False
    if strip_text[0] in ['-', '+']:
        if len(strip_text) > 1:
            return strip_text.isdigit()
        else:
            return False
    else:
        return strip_text.isdigit()

def is_bool(text):
    if text.strip().lower() in ['true', 'false', 't', 'f']:
        return True
    else:
        return False

def is_polygon(text):
    text = text.strip().lower()
    if text == '':
        return False
    if text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    polygon_sub_str = 'multipolygon'
    if text.startswith(polygon_sub_str):
        text = text.replace(polygon_sub_str, '')
        for ch in text:
            if not (ch in ['(', ')', ',' , ' ', '.'] or ch.isdigit()):
                return False
        return True
    else:
        return False

def is_prime(N):
    if N <= 1:
        return False
    sqrt_num = int(math.sqrt(N))
    for i in range(2, sqrt_num + 1):
        if N % i == 0:
            return False
    return True

def norm_text(text):
    if text is None:
        return ''
    return text.strip().lower()

def infer_col_type(table_data, infer_cols=None, infer_rows=None):
    col_data = table_data['columns']
    row_data = table_data['rows']

    if infer_cols is None:
        cols_used = list(range(len(col_data)))
    else:
        cols_used = infer_cols
    
    if infer_rows is None:
        rows_used = list(range(len(row_data)))
    else:
        rows_used = infer_rows

    for col in cols_used:
        col_info = col_data[col]
        type_lst = []
        bool_count = 0
        int_count = 0
        float_count = 0
        polygon_count = 0
        for row in rows_used:
            row_item = row_data[row]
            if (bool_count >= 3) or (int_count >= 3) or (float_count >= 3) or (polygon_count >= 1):
                break
            cell_info = row_item['cells'][col]
            cell_text = cell_info['text']
            infer_type = CellDataType.Other
            if (cell_text != ''):
                if is_bool(cell_text):
                    infer_type = CellDataType.BOOL
                    bool_count += 1
                elif is_int(cell_text):
                    infer_type = CellDataType.INT
                    int_count += 1
                elif is_float(cell_text):
                    infer_type = CellDataType.FLOAT
                    float_count += 1
                elif is_polygon(cell_text):
                    infer_type = CellDataType.POLYGON
                    polygon_count += 1
                else:
                    infer_type = CellDataType.Other
                cell_info['infer_type'] = infer_type
                type_lst.append(infer_type)

        if len(type_lst) > 0:
            type_arr = np.array(type_lst)
            if bool_count >= 3:
                col_info['infer_type'] = CellDataType.BOOL
            elif int_count >= 3: 
                col_info['infer_type'] = CellDataType.INT
            elif float_count >= 3: 
                col_info['infer_type'] = CellDataType.FLOAT
            elif polygon_count >= 1:
                col_info['infer_type'] = CellDataType.POLYGON
            else:
                col_info['infer_type'] = CellDataType.Other
                
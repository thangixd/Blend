import os
import json
import random
import numpy as np
import glob
import util
import llm
import uuid
import gpt
import sql_parser
from constant import GenConstant

class PredOP:
    eq = '='
    greater = '>'
    less = '<'
    between = 'between'

class CtrlProb:
    def __init__(self, aggr=0.5, group_by=0.2, order_by=0.2, having=0.8):
        self.aggr = aggr
        self.group_by = group_by
        self.order_by = order_by
        self.having = having

class QuestionGenerator:
    def __init__(self, dataset, prompt_dir):
        self.buffer = []
        self.q_size_per_table = 10
        self.prompt_dir = prompt_dir
        self.table_sep = ' | ' 
        self.init_prompt_tags()
        self.init_messages()
        self.log_dir = os.path.join(prompt_dir, 'log', dataset)
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        self.numeric_aggr_op_lst = ['max', 'min', 'avg', 'sum']
        self.text_aggr_op_lst = ['count']
        self.pred_op_lst = [PredOP.eq, PredOP.greater, PredOP.less, PredOP.between]
        self.ctrl_prob = CtrlProb()

    def raise_not_implemented(self):
        raise ValueError('Not implemented')

    def init_log_setting(self):
        self.raise_not_implemented()

    def init_prompt_tags(self):
        self.prompt_caption_tag = 'Table Caption:'
        self.prompt_cell_tag = 'Table Data:'
        self.prompt_sql_tag = '\nSQLs:'
    
    def read_prompt(self, name):
        prompt_file = os.path.join(self.prompt_dir, name + '.pmt')
        with open(prompt_file) as f:
                prompt_text = f.read()
        return prompt_text

    def init_messages(self):
        return

    def get_prompt_cols(self, table_data):
        prompt_cols = table_data['prompt_cols']
        return prompt_cols
    
    def get_prompt_rows(self, table_data):
        prompt_rows = table_data['prompt_rows']
        return prompt_rows

    def get_col_header_prompt(self, table_data):
        col_data = table_data['columns']
        prompt_cols = self.get_prompt_cols(table_data)
        text_lst = [col_data[col]['text'] for col in prompt_cols] 
        header_prompt = self.table_sep.join(text_lst)
        return header_prompt
            
    def col_data_complete(self, row_item, col_lst):
        for col in col_lst:
            cell_text = row_item['cells'][col]['text'].strip() 
            if cell_text == '':
                return False
        return True

    def use_table_title(self, sel_col_lst, cond_info_lst):
        cond_col_lst = [a['col'] for a in cond_info_lst]
        num_cols = len(set(sel_col_lst + cond_col_lst))
        opt = 0
        if num_cols > 0:
            title_prob = 0.9 / num_cols
            opt = np.random.choice([0, 1], size=1, p=[1-title_prob, title_prob])[0]
        return int(opt)

    def get_numeric_cols(self, col_data, col_lst):
        numeric_col_lst = []
        for col in col_lst:
            infer_type = col_data[col].get('infer_type', None)
            if infer_type in [util.CellDataType.INT, util.CellDataType.FLOAT]:
                numeric_col_lst.append(col)
        return numeric_col_lst

    def sample_col_numerics(self, col_data, col, row_data, sample_size=1, sort=False):
        sample_rows_data = random.sample(row_data, min(100, len(row_data)))
        numeric_lst = []
        all_int = True
        for row_item in sample_rows_data:
            cell_info = row_item['cells'][col]
            val_text = cell_info['text'].strip()
            if val_text == '':
                continue
            if util.is_int(val_text):
                numeric_lst.append(int(val_text))
            else:
                if util.is_float(val_text):
                    numeric_lst.append(float(val_text))
                    all_int = False
        if len(numeric_lst) == 0:
            return None
        val_lst = np.random.uniform(low=min(numeric_lst), 
                                    high=max(numeric_lst), 
                                    size=(sample_size,))
        if sort:
            val_lst.sort()
        if all_int:
            val_lst = [int(val) for val in val_lst]
            str_val_lst = [str(int(val)) for val in val_lst]
        else:
            str_val_lst = ["{:.2f}".format(val) for val in val_lst]        
        return str_val_lst

    def use_aggr_select(self):
        prop = self.ctrl_prob.aggr
        prop_lst = [1-prop, prop]
        opt = np.random.choice([0, 1], size=1, p=prop_lst)[0]
        return int(opt)

    def use_group_by(self):
        prop_group_by = self.ctrl_prob.group_by
        opt_group_by = np.random.choice([0, 1], size=1, p=[1-prop_group_by, prop_group_by])[0]
        return int(opt_group_by)

    def process_no_aggr_select(self, col_data, col_lst):
        if len(col_lst) == 0:
            return []
        sel_col_num_lst = [1, 2]
        sel_col_num = random.sample(sel_col_num_lst, 1)[0]
        sample_size = min(len(col_lst), sel_col_num)
        sel_col_lst = random.sample(col_lst, sample_size)
        select_struct = [{'col':col, 'col_name':col_data[col]['text'].strip()} for col in sel_col_lst]
        return select_struct
    
    def sampel_predicate_cols(self, col_data, col_lst, sel_cols, sql_struct):
        if ('order_by' in sql_struct) or ('group_by' in sql_struct):
            col_num_lst = [0, 1, 2]
            prob_lst = [0.1, 0.45, 0.45]
        else:
            col_num_lst = [1, 2]
            prob_lst = [0.5, 0.5]
        num_pred_cols = np.random.choice(col_num_lst, size=1, p=prob_lst)[0]
        pred_cols_all = [a for a in col_lst if a not in sel_cols]
        sample_size = min(num_pred_cols, len(pred_cols_all))
        pred_col_lst = random.sample(pred_cols_all, sample_size)
        return pred_col_lst

    def use_order_by(self):
        prop = self.ctrl_prob.order_by
        opt = np.random.choice([0, 1], size=1, p=[1-prop, prop])[0]
        return int(opt)

    def proceess_order_by_limit(self, col_data, numeric_col_lst):
        order_by_col = random.sample(numeric_col_lst, 1)[0]
        direction = random.sample(['desc', 'asc'], 1)[0]
        top_k = int(np.random.choice([3, 5], size=1)[0])
        order_by_info = {
            'col':order_by_col,
            'col_name':col_data[order_by_col]['text'].strip(),
            'direction':direction
        }
        limit_info = {'top':top_k}
        return order_by_info, limit_info

    def sample_predicate_row(self, col_data, pred_cols, row_data, row_lst):
        qualified_row_lst = []
        for row in row_lst:
            row_item = row_data[row]
            if not self.col_data_complete(row_item, pred_cols):
                continue
            qualified_row_lst.append(row)
        if len(qualified_row_lst) == 0:
            return None
        sampel_row = random.sample(qualified_row_lst, 1)[0]
        return sampel_row
    
    def process_aggr_select(self, col_data, col_lst, numeric_col_lst, row_data):
        aggr_col = random.sample(numeric_col_lst, 1)[0]
        aggr_op = random.sample(self.numeric_aggr_op_lst, 1)[0]

        select_column_lst = []
        aggr_col_info = {}
        aggr_col_info['col'] = aggr_col
        aggr_col_info['col_name'] = col_data[aggr_col]['text'].strip()
        aggr_col_info['aggr'] = aggr_op

        select_column_lst.append(aggr_col_info)

        group_by_info = None
        having_cond = None
        text_col_lst = list(set(col_lst) - set(numeric_col_lst))
        opt_group_by = int(0)
        if len(text_col_lst) > 0:
            opt_group_by = self.use_group_by()
        if opt_group_by:
            #choose the group by col
            group_by_info = {}
            
            group_by_col = random.sample(text_col_lst, 1)[0]
            group_by_info['col'] = group_by_col
            group_by_info['col_name'] = col_data[group_by_col]['text'].strip()

            select_column_lst.insert(0, group_by_info)

            prop_having = self.ctrl_prob.having
            opt_having = int(np.random.choice([0, 1], size=1, p=[1-prop_having, prop_having])[0])
            if opt_having:
                having_op_lst = [a for a in self.pred_op_lst if a != PredOP.eq]
                pred_op = random.sample(having_op_lst, 1)[0]
                str_numeric = None
                str_numeric_2 = None
                if pred_op == PredOP.between:
                    sample_numeric_lst = self.sample_col_numerics(col_data, aggr_col, row_data,
                                            sample_size=2, sort=True)
                    str_numeric = sample_numeric_lst[0]
                    str_numeric_2 = sample_numeric_lst[1]
                else:
                    sample_numeric_lst = self.sample_col_numerics(col_data, aggr_col, row_data)
                    str_numeric = sample_numeric_lst[0]

                having_cond = {'col':aggr_col,
                                'col_name':col_data[aggr_col]['text'].strip(),
                                'aggr':aggr_op,
                                'pred_op':pred_op,
                                'val':str_numeric
                            }
                if pred_op == PredOP.between:
                    having_cond['val_2'] = str_numeric_2
                    
        else:
            #There is no group by.
            opt_more_aggr_col = random.sample([0, 1], 1)[0]
            if opt_more_aggr_col:
                more_col_space = [a for a in numeric_col_lst if a != aggr_col]
                if len(more_col_space) > 0:
                    aggr_col_2_info = {}
                    aggr_col_2 = random.sample(more_col_space, 1)[0]
                    aggr_op_2 = random.sample(self.numeric_aggr_op_lst, 1)[0]
                    aggr_col_2_info['col'] = aggr_col_2
                    aggr_col_2_info['col_name'] = col_data[aggr_col_2]['text'].strip()
                    aggr_col_2_info['aggr'] = aggr_op_2
                    select_column_lst.append(aggr_col_2_info)

        return select_column_lst, group_by_info, having_cond

    def sample_sql(self, table_data, sample_size):
        table_caption = table_data['documentTitle'].strip()
        row_data = table_data['rows']
        row_lst = self.get_prompt_rows(table_data)
        col_data = table_data['columns']
        col_lst = self.get_prompt_cols(table_data)
        
        sql_info_lst = []
        max_try = sample_size * 10
        num_try = 0
        while (len(sql_info_lst) < sample_size):
            if num_try > max_try:
                break
            num_try += 1
            numeric_col_lst = self.get_numeric_cols(col_data, col_lst)
            if len(numeric_col_lst) > 0:
                opt_aggr_select = self.use_aggr_select()
            else:
                opt_aggr_select = False
            sql_struct = {'options':{}}
            sel_col_lst = []
            if opt_aggr_select:
                aggr_select_info = self.process_aggr_select(col_data, col_lst, 
                                   numeric_col_lst, row_data)
                sql_struct['select'] = aggr_select_info[0]
                sql_struct['group_by'] = aggr_select_info[1]
                sql_struct['having'] = aggr_select_info[2]
            else:
                sql_struct['select'] = self.process_no_aggr_select(col_data, col_lst)
                opt_order_by = int(0)
                if len(numeric_col_lst) > 0:
                    opt_order_by = self.use_order_by()
                if opt_order_by:
                    order_by_info, limit_info = self.proceess_order_by_limit(col_data, numeric_col_lst)
                    sql_struct['order_by'] = order_by_info
                    sql_struct['limit'] = limit_info
        
            sel_column_info = sql_struct['select']
            sel_col_lst = [a['col'] for a in sel_column_info]
            pred_col_lst = self.sampel_predicate_cols(col_data, col_lst, sel_col_lst, sql_struct)
            cond_lst = self.construct_predicates(pred_col_lst, col_data, 
                                  row_data, row_lst)
            sql_struct['where'] = cond_lst
            
            opt_use_title = int(0)
            if table_caption != '':
                opt_use_title = self.use_table_title(sel_col_lst, cond_lst)
            sql_struct['options']['use_title'] = opt_use_title
            meta = {
                'table_id':table_data['tableId'],
                'title':table_data['documentTitle'].strip(),
                'sql_struct':sql_struct
            }
            sql_prompt, ref_col_name_set, ref_val_set = self.struct_to_sql(sql_struct, col_data)
            meta['ref_col_names'] = list(ref_col_name_set)
            meta['ref_values'] = list(ref_val_set)
            sql_info = {'id':str(uuid.uuid4()), 'sql':sql_prompt, 'meta':meta}
            sql_info_lst.append(sql_info)
        
        return sql_info_lst

    def group_by_to_prompt(self, group_by_struct):
        return

    def sql_wrap_column(self, text, ref_set):
        ref_set.add(util.norm_text(text))
        wrap_text = '"' + text + '"'
        return wrap_text

    def sql_wrap_value(self, text, infer_type=None, ref_set=None):
        if infer_type == util.CellDataType.Other:
            ref_set.add(util.norm_text(text))
        wrap_text = '`' + text + '`'
        return wrap_text

    def select_to_sql(self, sql_struct, col_name_set):
        sql_text = 'select'
        select_struct = sql_struct['select']
        for offset, col_info in enumerate(select_struct):
            if offset > 0:
                sql_text += ','
            aggr = col_info.get('aggr', None)
            col_name = self.sql_wrap_column(col_info['col_name'], col_name_set)
            if aggr is None:
                sql_text += f' {col_name}'
            else:
                sql_text += f' {aggr}({col_name})'
        return sql_text

    def where_to_sql(self, sql_struct, col_data, col_name_set, val_set):
        where_struct = sql_struct.get('where', None)
        if (where_struct is None) or (len(where_struct) == 0):
            return ''
        sql_text = ' where'
        for offset, cond_item in enumerate(where_struct):
            if offset > 0:
                sql_text += ' and'
            col_name = self.sql_wrap_column(cond_item['col_name'], col_name_set)
            op = cond_item['op']
            col_infer_type = col_data[cond_item['col']]['infer_type']
            val = self.sql_wrap_value(cond_item['val'], col_infer_type, val_set)
            if op == PredOP.between:
                val_2 = self.sql_wrap_value(cond_item['val_2'])
                sql_text += f' {col_name} between {val} and {val_2}'
            else:
                sql_text += f' {col_name} {op} {val}'
        return sql_text

    def group_by_to_sql(self, sql_struct, col_name_set):
        group_by = sql_struct.get('group_by', None)
        if group_by is None:
            return ''
        
        col_name = self.sql_wrap_column(group_by['col_name'], col_name_set)
        sql_text = f'group by {col_name}'
        return sql_text

    def having_to_sql(self, sql_struct, col_name_set, val_set):
        having = sql_struct.get('having', None)
        if having is None:
            return ''
        aggr = having['aggr']
        col_name = self.sql_wrap_column(having['col_name'], col_name_set)
        pred_op = having['pred_op']
        val = self.sql_wrap_value(having['val'], val_set)
        if pred_op == 'between':
            val_2 = self.sql_wrap_value(having['val_2'], val_set)
            sql_text = f'having {aggr}({col_name}) between {val} and {val_2}'
        else:
            sql_text = f'having {aggr}({col_name}) {pred_op} {val}'
        return sql_text

    def order_by_to_sql(self, sql_struct, col_name_set):
        order_by = sql_struct.get('order_by', None)
        if order_by is None:
            return ''
        
        col_name = self.sql_wrap_column(order_by['col_name'], col_name_set)
        direction = order_by['direction']
        sql_text = f'order by {col_name} {direction}'
        return sql_text

    def limit_to_sql(self, sql_struct):
        limit = sql_struct.get('limit', None)
        if limit is None:
            return ''
        top = limit['top']
        sql_text = f'limit {top}'
        return sql_text

    def struct_to_sql(self, sql_struct, col_data):
        ref_col_name_set = set()
        ref_val_set = set() # only non-numeric

        select_sql = self.select_to_sql(sql_struct, ref_col_name_set)
        where_sql = self.where_to_sql(sql_struct, col_data, ref_col_name_set, ref_val_set)
        group_by_sql = self.group_by_to_sql(sql_struct, ref_col_name_set)
        having_sql = self.having_to_sql(sql_struct, ref_col_name_set, ref_val_set)
        order_by_sql = self.order_by_to_sql(sql_struct, ref_col_name_set)
        limit_sql = self.limit_to_sql(sql_struct)

        sql = select_sql
        if where_sql != '':
            sql += ' ' + where_sql
        if group_by_sql != '':
            sql += ' ' + group_by_sql
        if having_sql != '':
            sql += ' ' + having_sql
        if order_by_sql != '':
            sql += ' ' + order_by_sql
        if limit_sql != '':
            sql += ' ' + limit_sql

        return sql, ref_col_name_set, ref_val_set

    def construct_predicates(self, where_cols, col_data, row_data, row_lst):
        cond_lst = []
        for col in where_cols:
            sample_value_1 = None
            sample_value_2 = None
            col_type = col_data[col].get('infer_type', None)
            if col_type in [util.CellDataType.INT, util.CellDataType.FLOAT]:
                op = random.sample(self.pred_op_lst, 1)[0]
                if op in [PredOP.greater, PredOP.less]:
                    sample_value_1 = self.sample_col_numerics(col_data, col, row_data)[0]
                elif op == PredOP.between:
                    value_lst = self.sample_col_numerics(col_data, col, row_data,
                                                         sample_size=2, sort=True)
                    sample_value_1 = value_lst[0]
                    sample_value_2 = value_lst[1]
            else:
                op = PredOP.eq
            
            col_name = col_data[col]['text'].strip()
            cond_info = {'col':col, 'col_name':col_name, 'op':op,  
                        'val':sample_value_1}
            if op == PredOP.between:
                cond_info['val_2'] = sample_value_2
            cond_lst.append(cond_info)
        eq_cond_lst = [a for a in cond_lst if a['op'] == PredOP.eq]
        if len(eq_cond_lst) > 0:
            eq_pred_cols = [a['col'] for a in eq_cond_lst]
            sample_row = self.sample_predicate_row(col_data, eq_pred_cols, row_data, row_lst)
            if sample_row is None:
                return []
            row_item = row_data[sample_row]
            for eq_cond in eq_cond_lst:
                pred_col = eq_cond['col']
                row_cells = row_item['cells']
                eq_cond['val'] = row_cells[pred_col]['text'].strip()
                eq_cond['row'] = sample_row
        return cond_lst
    
    def find_too_large_cols(self, table_data, col_lst):
        row_data = table_data['rows']
        sample_size = min(len(row_data), 30)
        ssample_row_data = random.sample(row_data, sample_size)
        too_large_col_lst = []
        for col in col_lst:
            size_lst = []
            for row_item in ssample_row_data:
                cell_lst = row_item['cells']
                char_size = len(cell_lst[col]['text'].strip())
                size_lst.append(char_size)
            mean_size = np.mean(size_lst)
            if mean_size > 2000:
                table_id = table_data['tableId']
                print(f'Too large table={table_id} col={col}, char size={mean_size}')
                too_large_col_lst.append(col)
        return too_large_col_lst

    def sample_prompt_data(self, table_data):
        col_data = table_data['columns']
        col_lst = list(range(len(col_data)))
        M = 50
        if len(col_lst) > M:
            Num_First = 5
            sample_cols = col_lst[:Num_First] + random.sample(col_lst[Num_First:], M - Num_First)
        else:
            sample_cols = col_lst
        too_large_col_set = set(self.find_too_large_cols(table_data, sample_cols))
        sample_cols_used = [a for a in sample_cols if a not in too_large_col_set]
        table_data['prompt_cols'] = sample_cols_used
        row_lst = list(range(len(table_data['rows'])))
        N = 10
        if len(row_lst) > N:
            sample_rows = random.sample(row_lst, N)
        else:
            sample_rows = row_lst
        table_data['prompt_rows'] = sample_rows

    def get_row_prompts(self, table_data):
        prompt_lst = []
        row_data = table_data['rows']
        prompt_cols = self.get_prompt_cols(table_data)
        prompt_rows = self.get_prompt_rows(table_data)
        for row in prompt_rows:
            row_item = row_data[row]
            cell_lst = row_item['cells'] 
            row_prompt = '\n' + self.table_sep.join([cell_lst[a]['text'] for a in prompt_cols])
            prompt_lst.append(row_prompt)
        return prompt_lst

    def get_sql_prompts(self, table_data):
        util.infer_col_type(table_data,
                            infer_cols=self.get_prompt_cols(table_data), 
                            infer_rows=self.get_prompt_rows(table_data)
                            )
        sql_info_lst = self.sample_sql(table_data, sample_size=self.q_size_per_table) 
        prompt_sql_info_lst = [] # may exceed size limit (not implemented now), so use another list
        row_data = table_data['rows']
        prompt_cols = self.get_prompt_cols(table_data)
        for sql_offset, sql_info in enumerate(sql_info_lst):
            sql_no = sql_offset + 1
            sql_text = sql_info['sql']
            sql_prompt = f'\n{sql_no}. {sql_text}'
            sql_info['sql_prompt'] = sql_prompt
            prompt_sql_info_lst.append(sql_info)
        return prompt_sql_info_lst

    def prompt_table_data(self, table_data):
        prompt = ''
        title = table_data['documentTitle'].strip()
        if title != '':
            #Add table caption
            prompt = self.prompt_caption_tag + '\n' + table_data['documentTitle']
        #Add cell data tag
        prompt += '\n' + self.prompt_cell_tag
        header_prompt = self.get_col_header_prompt(table_data)
        prompt += '\n' + header_prompt
        row_prompt_lst = self.get_row_prompts(table_data)
        for row_prompt in row_prompt_lst:
            prompt += row_prompt
        return prompt

    def table_title_to_ner(self, title):
        ner_lst = []
        prompt = self.read_prompt('ner')
        prompt = prompt.replace('{INPUT_TEXT}', title)
        
        self.messages[-1]['content'] = prompt
        response = gpt.chat_complete(self.client, self.messages, 'title_to_ner')

        TAG_BEGIN = 'Text_1_NER_BEGIN'
        TAG_END = 'Text_1_NER_END'
        pos_1 = response.find(TAG_BEGIN)
        if pos_1 < 0:
            return ner_lst
        pos_2 = response.find(TAG_END, pos_1 + len(TAG_BEGIN))
        if pos_2 < 0:
            return ner_lst
        entity_text = response[pos_1 + len(TAG_BEGIN) : pos_2]
        out_text_lst = entity_text.split('\n')
        for line in out_text_lst:
            item = line.split('||')
            if len(item) == 3:
                entity = item[1].strip()
                ner_lst.append(entity)
        return ner_lst

    def apply_title_to_sql(self, title, sql_using_title_lst):
        MAX_TRY = 3
        ner_set = set()
        for _ in range(MAX_TRY):
            ner_lst = self.table_title_to_ner(title)
            ner_set.update(set(ner_lst))
            if len(ner_set) > 0:
                break

        ner_complete_lst = list(ner_set)
        for sql_info in sql_using_title_lst:
            sql_info['meta']['title_ner'] = ner_complete_lst
            if len(ner_complete_lst) > 0:
                sample_entity = random.sample(ner_complete_lst, 1)[0]
                sql_info['meta']['sample_entity'] = sample_entity
                sql_info['sql_prompt'] += ' , where context : { name : null , operator : like , value : ' + sample_entity + ' }'
        
    def prompt_sql_to_question(self, table_data):
        table_prompt = self.prompt_table_data(table_data)
        sql2quest_prompt = self.read_prompt('sql2question')
        sql2quest_prompt += '\n' + table_prompt
        sql2quest_prompt += self.prompt_sql_tag
        sql_info_lst = self.get_sql_prompts(table_data)
        
        sql_using_title_lst = [a for a in sql_info_lst if a['meta']['sql_struct']['options']['use_title']]
        if len(sql_using_title_lst) > 0:
            self.apply_title_to_sql(table_data['documentTitle'], sql_using_title_lst)

        for sql_info in sql_info_lst:
            sql2quest_prompt += sql_info['sql_prompt']
        return sql2quest_prompt, table_prompt, sql_info_lst

    def write_sql_log(self, sql_info_lst):
        sql_info_file = os.path.join(self.log_dir, f'sql_info_{self.time_stamp}.jsonl')
        with open(sql_info_file, 'w') as f_o:
            for sql_info in sql_info_lst:
                f_o.write(json.dumps(sql_info) + '\n')
    
    def write_response_log(self, response):
        out_file = os.path.join(self.log_dir, f'response_{self.time_stamp}.txt')
        with open(out_file, 'w') as f_o:
            f_o.write(response)

    def prompt_copied_questions(self, sql_info_lst):
        copied_seq_no = 0
        copied_sql_info_lst = []
        for sql_info in sql_info_lst:
            question = sql_info['question']
            if question is None:
                continue
            copied_col_lst, copied_cell_lst = self.check_copy_text(sql_info)
            copied_text_lst = copied_col_lst + copied_cell_lst
            if len(copied_text_lst) == 0:
                continue
            paraphrase_opt = random.sample([True,False],1)[0]
            if not paraphrase_opt:
                continue
            copied_seq_no += 1
            copied_sub_text = ''
            for copied_text in copied_text_lst:
                copied_sub_text += ' ` ' + copied_text + ' ` ,'
            copied_sub_text = copied_sub_text[:-1]

            no_copy_prompt = f'\n{copied_seq_no}. Explain {copied_sub_text}' 
            no_copy_prompt += f' and then paraphrase question : ` {question} `'
            no_copy_prompt += ' Using different wording than ' + copied_sub_text + ' .'

            sql_info['no_copy_prompt'] = no_copy_prompt
            copied_sql_info_lst.append(sql_info)
        return copied_sql_info_lst

    def clear_cycle_check_tag(self, sql_info):
        to_del_keys = ['consistent_col', 'back_meta', 'back_error', 'consistent']
        for key in to_del_keys:
            if key in sql_info:
                del sql_info[key]

    def check_question_from_sql_consistency(self, table_prompt, sql_info_lst):
        for sql_info in sql_info_lst:
            sql_info['question'] = sql_info[GenConstant.Q_From_SQL_1]
        self.cycle_check(table_prompt, sql_info_lst)
        cst_info_lst = []
        for sql_info in sql_info_lst:
            if sql_info['consistent']:
                self.clear_cycle_check_tag(sql_info)
                cst_info_lst.append(sql_info)
        return cst_info_lst

    def generate_questions(self, table_data, ctrl_prob=None):
        if ctrl_prob is not None:
            self.ctrl_prob = ctrl_prob
        self.init_log_setting()
        self.sample_prompt_data(table_data)
        sql2quest_prompt, table_prompt, sql_info_lst = self.prompt_sql_to_question(table_data)
        self.sql_to_question(sql2quest_prompt, sql_info_lst)

        #Cycle_Check question_from_sql_1 (the direct translation)
        sql_info_lst = self.check_question_from_sql_consistency(table_prompt, sql_info_lst)
        
        for sql_info in sql_info_lst:
            #Use the paraphrased one. 
            sql_info['question'] = sql_info[GenConstant.Q_From_SQL_2]

        copied_sql_info_lst = self.prompt_copied_questions(sql_info_lst)
        if len(copied_sql_info_lst) > 0:
            self.rewrite_question_copied_text(table_prompt, copied_sql_info_lst)
        
        self.cycle_check(table_prompt, sql_info_lst)
        question_lst = []
        for sql_info in sql_info_lst:
            if not sql_info['consistent']:
                continue
            q_text = sql_info['question']
            q_info = {
                'id':sql_info['id'],
                'question':q_text,
                'question_from_sql_1':sql_info[GenConstant.Q_From_SQL_1],
                'meta':sql_info['meta']
            }
            question_lst.append(q_info)
        return question_lst
    
    def rewrite_question_copied_text(self, table_prompt, copied_sql_info_lst):
        prompt = self.read_prompt('no_copy_text')
        prompt = prompt.replace('{Table_Data}', table_prompt)
        
        question_part = ''
        for sql_info in copied_sql_info_lst:
            question_part += sql_info['no_copy_prompt']
        prompt = prompt.replace('{Questions}', question_part)
        
        self.messages[-1]['content'] = prompt
        response = gpt.chat_complete(self.client, self.messages, 'rewrite_copied')
        out_text_lst = response.split('\n')
        start_pos = 0
        for offset, sql_info in enumerate(copied_sql_info_lst):
            q_no = offset + 1
            tag_begin = f'Paraphrased_Begin_{q_no}:'
            tag_end = f'Paraphrased_End_{q_no}'
            pos_1 = response.find(tag_begin, start_pos)
            pos_2 = response.find(tag_end, start_pos)
            if pos_1 >= 0 and pos_2 > pos_1:
                q_text = response[pos_1+len(tag_begin):pos_2].strip()
                sql_info = copied_sql_info_lst[q_no - 1]
                sql_info['question'] = q_text
                start_pos = pos_2 + 1
    
    def is_sub_text(self, text_1, text_2):
        pos = text_2.find(text_1)
        if pos < 0:
            return False
        if (pos > 0) and (text_2[pos-1] in [' ', '(']):
            pos_2 = pos + len(text_1)
            if (pos_2 < len(text_2)) and (text_2[pos_2] in [' ', ')']):
                return True
        return False

    def check_copy_text(self, sql_info):
        question = sql_info['question']
        meta_info = sql_info['meta']
        col_name_lst = meta_info['ref_col_names']
        question_text = ' '.join(util.norm_text(question).split())
        copied_col_lst = []
        for col_name in col_name_lst:
            col_text = ' '.join(col_name.split())
            if self.is_sub_text(col_text, question_text):
                copied_col_lst.append(col_name)
        
        copied_cell_lst = []
        cell_value_lst = meta_info['ref_values']
        for cell_value in cell_value_lst:
            cell_words = cell_value.split()
            if len(cell_words) >= 3:
                cell_text = ' '.join(cell_words)
                cell_text = ' '.join(cell_words)
                if self.is_sub_text(cell_text, question_text):
                    copied_cell_lst.append(cell_value)

        return copied_col_lst, copied_cell_lst

    def cycle_check(self, table_prompt, sql_info_lst):
        self.cycle_check_col(table_prompt, sql_info_lst)

        sql_to_check_title_lst = []
        for sql_info in sql_info_lst:
            if sql_info['consistent_col'] and sql_info['meta']['sql_struct']['options']['use_title']:
                sql_to_check_title_lst.append(sql_info)
        
        if len(sql_to_check_title_lst) > 0:
            self.cycle_check_title(table_prompt, sql_to_check_title_lst)

        for sql_info in sql_info_lst:
            if not sql_info['meta']['sql_struct']['options']['use_title']:
                sql_info['consistent'] = sql_info['consistent_col']
            else:
                sql_info['consistent'] = (sql_info['consistent_col'] and sql_info['consistent_title'])

    def cycle_check_title(self, table_prompt, sql_info_lst):
        prompt = self.read_prompt('cycle_check_title')
        prompt += '\n' + table_prompt + '\n'
        prompt += '\nQuestions:'
        seq_no = 0
        pmt_row_count = 0
        for sql_info in sql_info_lst:
            sql_info['consistent_title'] = None
            seq_no += 1

            if 'sample_entity' not in sql_info['meta']:
                #print('No sample entity in SQL : ', sql_info)
                continue
            sample_entity = sql_info['meta']['sample_entity']
            back_meta = sql_info['back_meta']
            more_cols = back_meta.get('more_cols', None)
            if more_cols is not None:
                prompt += f'\n{seq_no}. Is ` ' + sample_entity + ' ` in the domain of ' + more_cols[0] + ' ?'
            else:
                prompt += f'\n{seq_no}. Is ` ' + sample_entity + ' ` an entity in the table caption ?'
            pmt_row_count += 1

        if pmt_row_count == 0:
            return
        self.messages[-1]['content'] = prompt
        response = gpt.chat_complete(self.client, self.messages, 'cycle_check_title')
        out_text_lst = response.split('\n')
        tag = 'Answer:'
        for line in out_text_lst:
            awr_pos = line.find(tag)
            if awr_pos < 0:
                continue
            q_no_pos = line.find('.', 0, awr_pos)
            if q_no_pos < 0:
                continue
            awr_text = line[awr_pos + len(tag):]
            answer = util.norm_text(awr_text)
            if answer not in ['yes', 'no']:
                continue
            q_no = int(line[:q_no_pos])
            item_sql_info = sql_info_lst[q_no - 1]
            item_sql_info['consistent_title'] = (True if answer == 'yes' else False)
        
    def cycle_check_col(self, table_prompt, sql_info_lst):
        prompt = self.read_prompt('cycle_check_col')
        prompt += '\n' + table_prompt
        prompt += '\n' + 'Questions:'
        
        sql_lst_to_check = []
        for sql_info in sql_info_lst:
            sql_info['consistent_col'] = None
            q_text = sql_info.get('question', None)
            if q_text is None:
                continue
            sql_lst_to_check.append(sql_info)
            seq_no  = len(sql_lst_to_check)
            prompt += f'\n{seq_no}. ' + q_text
        
        if len(sql_lst_to_check) == 0:
            return
        prompt += "\n\nLet's think step by step to output SQL:"
        self.messages[-1]['content'] = prompt
        response = gpt.chat_complete(self.client, self.messages, 'cycle_check_col')

        #self.write_sql_log(sql_lst_to_check)
        #self.write_response_log(response)
        #import pdb; pdb.set_trace()
        self.check_back_sql(sql_lst_to_check, response)
        
    def check_back_sql(self, sql_lst_to_check, response):
        tag_start = 'SQL_START_'
        tag_end = 'SQL_SEP_'
        start_pos = 0
        for offset, sql_info in enumerate(sql_lst_to_check):
            sql_no = offset + 1
            tag_1 = tag_start + str(sql_no)
            tag_2 = tag_end + str(sql_no)
            pos_1 = response.find(tag_1, start_pos)
            if pos_1 < 0:
                continue
            pos_2 = response.find(tag_2, pos_1 + len(tag_1))
            if pos_2 < 0:
                continue
            back_sql = response[pos_1 + len(tag_1):pos_2]
            #print(sql_no, '. ', sql_info['question'])
            #print(sql_no, '. ', back_sql)
            consistent = self.compare_sql_meta(sql_info, back_sql)
            #print(sql_no, '. ', 'consistent=', consistent)
            sql_info['consistent_col'] = consistent
            start_pos = pos_2 + len(tag_2)

    def get_cmp_select_info(self, select_struct):
        cmp_struct_lst = []
        for item in select_struct:
            col_name = util.norm_text(item.get('col_name', None))
            aggr = util.norm_text(item.get('aggr', None))
            cmp_item = {
                'col_name':col_name,
                'aggr':aggr
            }
            cmp_struct_lst.append(cmp_item)
        sorted_cmp_struct_lst = sorted(cmp_struct_lst,key=lambda x:x['col_name'])
        return sorted_cmp_struct_lst

    def compare_select(self, sql_struct, back_stmt, error):
        select = sql_struct.get('select', None)
        back_select = sql_parser.get_select(back_stmt)
        cmp_select = self.get_cmp_select_info(select)
        cmp_back_select = self.get_cmp_select_info(back_select)
        same = (cmp_select == cmp_back_select)
        if not same:
            error['select'] = True
        return same

    def get_cmp_where_info(self, where_struct):
        cmp_struct_lst = []
        for cond_info in where_struct:
            cmp_item = {
                'col_name':util.norm_text(cond_info.get('col_name', None)),
                'op':util.norm_text(cond_info.get('op', None)),
                'val':util.norm_text(cond_info.get('val', None)),
                'val_2':util.norm_text(cond_info.get('val_2', None))
            }
            cmp_struct_lst.append(cmp_item)
        
        sorted_cmp_struct_lst = sorted(cmp_struct_lst,key=lambda x:x['col_name'])
        return sorted_cmp_struct_lst

    def compare_where(self, sql_struct, back_stmt, back_meta, error):
        use_title = sql_struct['options']['use_title']
        where = sql_struct.get('where', None)
        back_where = sql_parser.get_where(back_stmt)
        cmp_where = self.get_cmp_where_info(where)
        cmp_back_where = self.get_cmp_where_info(back_where)
        same = False
        if use_title:
            if len(cmp_where) < len(cmp_back_where):
                cmp_back_where_updated = []
                sql_cond_col_names = [a['col_name'] for a in cmp_where]
                more_col_name_lst = []
                for back_cond in cmp_back_where:
                    if back_cond['col_name'] not in sql_cond_col_names:
                        more_col_name_lst.append(back_cond['col_name'])
                    else:
                        cmp_back_where_updated.append(back_cond)
                
                same = (cmp_where == cmp_back_where_updated)
                if same:
                    back_meta['more_cols'] = more_col_name_lst
        else:
            same = (cmp_where == cmp_back_where)

        if not same:
            error['where'] = True
        return same

    def get_cmp_group_by_info(self, group_struct):
        if group_struct is None:
            return None
        col_name = util.norm_text((group_struct.get('col_name', None)))
        cmp_info = {
            'col_name':col_name
        }
        return cmp_info

    def compare_group_by(self, sql_struct, back_stmt, error):
        group_by = sql_struct.get('group_by', None)
        back_group_by = sql_parser.get_group_by(back_stmt)
        cmp_group = self.get_cmp_group_by_info(group_by)
        cmp_back_group = self.get_cmp_group_by_info(back_group_by)
        same = (cmp_group == cmp_back_group)
        if not same:
            error['group_by'] = True
        return same

    def get_cmp_having_info(self, having_struct):
        if having_struct is None:
            return None
        cmp_info = {
            'col_name':util.norm_text((having_struct.get('col_name', None))),
            'aggr':util.norm_text((having_struct.get('aggr', None))),
            'pred_op':util.norm_text((having_struct.get('pred_op', None))),
            'val':util.norm_text((having_struct.get('val', None))),
            'val_2':util.norm_text((having_struct.get('val_2', None)))
        }
        return cmp_info

    def compare_having(self, sql_struct, back_stmt, error):
        having = sql_struct.get('having', None)
        back_having = sql_parser.get_having(back_stmt)
        cmp_having = self.get_cmp_having_info(having)
        cmp_back_having = self.get_cmp_having_info(back_having)
        same = (cmp_having == cmp_back_having)
        if not same:
            error['having'] = True
        return same

    def get_cmp_order_by_info(self, order_by_struct):
        if order_by_struct is None:
            return None
        cmp_info = {
            'col_name':util.norm_text(order_by_struct.get('col_name', None)),
            'direction':util.norm_text(order_by_struct.get('direction', None)),
        }
        return cmp_info

    def compare_order_by(self, sql_struct, back_stmt, error):
        order_by = sql_struct.get('order_by', None)
        back_order_by = sql_parser.get_order_by(back_stmt)
        cmp_order_by = self.get_cmp_order_by_info(order_by)
        cmp_back_order_by = self.get_cmp_order_by_info(back_order_by)
        same = (cmp_order_by == cmp_back_order_by)
        if not same:
            error['order_by'] = True
        return same

    def get_cmp_limit_info(self, limit_struct):
        if limit_struct is None:
            return None
        cmp_info = {
            'top':limit_struct.get('top')
        }
        return cmp_info

    def compare_limit(self, sql_struct, back_stmt, error):
        limit = sql_struct.get('limit', None)
        back_limit = sql_parser.get_limit(back_stmt)
        cmp_limit = self.get_cmp_limit_info(limit)
        cmp_back_limit = self.get_cmp_limit_info(back_limit)
        same = (cmp_limit == cmp_back_limit)
        if not same:
            error['limit'] = True
        return same

    def compare_sql_meta(self, sql_info, back_sql):
        try:
            back_stmt = sql_parser.parse_sql(back_sql)
        except:
            #grammar error
            return False
        sql_struct = sql_info['meta']['sql_struct']
        sql_info['back_meta'] = {}
        back_meta = sql_info['back_meta']
        error = {}
        sql_info['back_error'] = error
        #import pdb; pdb.set_trace()
        select_same = self.compare_select(sql_struct, back_stmt, error)
        where_same = self.compare_where(sql_struct, back_stmt, back_meta, error)
        group_by_same = self.compare_group_by(sql_struct, back_stmt, error)
        having_same = self.compare_having(sql_struct, back_stmt, error)
        order_by_same = self.compare_order_by(sql_struct, back_stmt, error)
        limit_same = self.compare_limit(sql_struct, back_stmt, error)

        all_same = (select_same and where_same and group_by_same and having_same and order_by_same and limit_same)
        return all_same
        
    def sql_to_question(self, sql2quest_prompt, sql_info_lst):
        self.raise_not_implemented()
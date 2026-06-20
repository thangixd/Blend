import os
from question_gen import QuestionGenerator
from datetime import datetime
import llm

class LlamaGenerator(QuestionGenerator):
    def __init__(self, dataset, prompt_dir):
        super().__init__(dataset, prompt_dir)
        self.llm_name = 'llama'
    
    def init_log_setting(self):
        now_t = str(datetime.now())
        self.time_stamp = now_t
        log_name = 'log_' + '_'.join(now_t.split()) + '.txt'
        llama_log_dir = os.path.join(self.log_dir, 'llama')
        if not os.path.isdir(llama_log_dir):
            os.makedirs(llama_log_dir)
        log_file = os.path.join(llama_log_dir, log_name)
        f_log = open(log_file, 'w')
        llm.set_logger(f_log)
    
    def read_prompt(self, name):
        prompt_file = os.path.join(self.prompt_dir, name + '.pmt')
        spec_name = name + '_' + self.llm_name
        spec_file = os.path.join(self.prompt_dir, spec_name + '.pmt')
        if os.path.isfile(spec_file):
            prompt_file = spec_file
        with open(prompt_file) as f:
                prompt_text = f.read()
        return prompt_text

    def sql_to_question(self, sql2quest_prompt, sql_info_lst):
        import pdb; pdb.set_trace()
        response = llm.query_llm(sql2quest_prompt)
        out_text_lst = response.split('\n')
        tag = 'Paraphrased_Begin:'
        for line in out_text_lst:
            quest_pos = line.find(tag)
            if quest_pos < 0:
                continue
            q_no_pos = line.find('.', 0, quest_pos)
            if q_no_pos < 0:
                continue
            q_no = int(line[:q_no_pos])
            pos = quest_pos + len(tag)
            question = line[pos:].strip()
            sql_info = sql_info_lst[q_no - 1]
            sql_info['question'] = question
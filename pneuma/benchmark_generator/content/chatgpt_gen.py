import os
import tiktoken
import gpt
from openai import OpenAI
from question_gen import QuestionGenerator
from datetime import datetime
from constant import GenConstant

class ChatgptGenerator(QuestionGenerator):

    def __init__(self, dataset, prompt_dir):
        self.token_encoding = tiktoken.encoding_for_model(gpt.MODEL_NAME)
        super().__init__(dataset, prompt_dir)
       
        output_size = 1000
        self.ctx_size = 4097 - output_size
        #set API key
        api_key = os.getenv('OPENAI_API_KEY', None)
        if api_key is None:
            raise ValueError('Need to set environment variable OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
    
    def init_messages(self):
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": None},
        ]
        #This is from openai cookbook on how to count tokens for chat completions API calls
        #https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
        tokens_per_message = 3
        num_tokens = 0
        for msg in self.messages:
            num_tokens += tokens_per_message
            for key, value in msg.items():
                if value is None:
                    continue
                num_tokens += len(self.token_encoding.encode(value))
    
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        self.num_meta_tokens = num_tokens

    def init_log_setting(self):
        now_t = str(datetime.now())
        self.time_stamp = now_t
        log_name = 'log_' + '_'.join(now_t.split()) + '.txt'
        gpt_log_dir = os.path.join(self.log_dir, 'chatgpt')
        if not os.path.isdir(gpt_log_dir):
            os.makedirs(gpt_log_dir)
        log_file = os.path.join(gpt_log_dir, log_name)
        f_log = open(log_file, 'w')
        gpt.set_logger(f_log)

    def sql_to_question(self, sql2quest_prompt, sql_info_lst):
        self.messages[-1]['content'] = sql2quest_prompt
        response = gpt.chat_complete(self.client, self.messages, 'sql_to_question')
        out_text_lst = response.split('\n')
        tag_1 = 'Translated_Begin:'
        tag_2 = 'Paraphrased_Begin:'

        tag_question_map = {}
        for line in out_text_lst:
            for tag in [tag_1, tag_2]:
                q_offset, question = self.get_question_from_tag(line, tag)
                if q_offset is None:
                    continue
                if q_offset not in tag_question_map:
                    tag_question_map[q_offset] = {}
                q_info = tag_question_map[q_offset]
                q_info[tag] = question

        
        for offset, sql_info in enumerate(sql_info_lst):
            sql_info[GenConstant.Q_From_SQL_1] = None
            sql_info[GenConstant.Q_From_SQL_2] = None
            if offset not in tag_question_map:
                continue
            q_info_map = tag_question_map[offset]
            if tag_1 in q_info_map:
                sql_info[GenConstant.Q_From_SQL_1] = q_info_map[tag_1]
            if tag_2 in q_info_map:
                sql_info[GenConstant.Q_From_SQL_2] = q_info_map[tag_2]

    
    def get_question_from_tag(self, line, tag):
        q_offset = None
        question = None
        quest_pos = line.find(tag)
        if quest_pos < 0:
            return q_offset, question
        q_no_pos = line.find('.', 0, quest_pos)
        if q_no_pos < 0:
            return q_offset, question
        q_no_str = line[:q_no_pos].strip()
        if not q_no_str.isdigit():
            return q_offset, question
        q_offset = int(q_no_str) - 1
        pos = quest_pos + len(tag)
        question = line[pos:].strip()
        return q_offset, question

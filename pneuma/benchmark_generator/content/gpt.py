import openai
import time

f_log = None
prompt_no = 0
MODEL_NAME = "gpt-4o-mini"

def set_key(key):
    openai.api_key = key

def set_logger(logger):
    global f_log
    f_log = logger

def write_log(log_msg, commit=False):
    f_log.write(log_msg + '\n')
    if commit:
        f_log.flush()

def chat_complete(client, messages, tag, temperature=0):
    global prompt_no
    prompt_no += 1
    write_log(f'prompt {tag}')
    write_log('-'*100)
    prompt = messages[-1]['content']
    write_log(prompt)
    retry_cnt = 0
    response = None
    wait_seconds = 3
    while response is None:
        try:
            #print('\n' + prompt)
            #print('-'*100)
            response = call_gpt(client, messages, temperature)
            #print(response)
            #input('\nNext ? ')
        except openai.RateLimitError as err:
            response = None
            retry_cnt += 1
            print('Error from GPT')
            print('\n')
            print(err)
            print('\n')
            print(f'Retry {retry_cnt} to call GPT in {wait_seconds} seconds\n')
            time.sleep(wait_seconds)
    
    write_log('\n' + '*'*30 + '\n')
    write_log(response)
    write_log('-'*100, commit=True)
    return response

def call_gpt(client, messages, temperature):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
    )
    out_msg = response.choices[0].message.content 
    return out_msg

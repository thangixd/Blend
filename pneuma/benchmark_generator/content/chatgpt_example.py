from openai import OpenAI
import gpt
import os

def read_input():
    with open('./prompt/example_2.pmt') as f:
        prompt = f.read()
    return prompt

def main():
    api_key = os.getenv('OPENAI_API_KEY', None)
    if api_key is None:
        raise ValueError('Need to set environment variable OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": None},
    ]

    messages[-1]['content'] = read_input()
    f_o_log = open('./prompt/log/chatgpt_log.txt', 'w')
    gpt.set_logger(f_o_log)
    print('prompt chatgpt')
    response = gpt.chat_complete(client, messages, 'example')
    print('ok')

if __name__ == '__main__':
    main()
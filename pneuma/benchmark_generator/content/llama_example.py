import llm

def example_1():
    prompt_file = './prompt/ner.pmt'
    with open(prompt_file) as f:
        prompt_text = f.read()
    text_lst_2 = [
    '1) 2001- 2013 Graduation Outcomes Borough- ALL STUDENTS,SWD,GENDER,ELL,ETHNICITY,EVER ELL',
    '2) 2006 - 2011  English Language Arts (ELA) Test Results by Grade - Citywide - by English Proficiency Status'
    ]
    text_lst = ['RealEstate2_4']
    prompt_text = prompt_text.replace('{NUM_TEXT}', str(len(text_lst)))
    prompt_text = prompt_text.replace('{LIST_OF_TEXT}', '\n'.join(text_lst))
    print(prompt_text)
    print('*' * 100)
    response = llm.query_llm(prompt_text)
    print(response)

def example_2():
    text_lst = [
    '2001- 2013 Graduation Outcomes Borough- ALL STUDENTS,SWD,GENDER,ELL,ETHNICITY,EVER ELL',
    '2006 - 2011  English Language Arts (ELA) Test Results by Grade - Citywide - by English Proficiency Status'
    ]

    prompt_text = 'Answer the questions below:'
    prompt_text += f'\n1. Is ` Graduation ` an entity in the text ` {text_lst[0]} ` ?'
    prompt_text += '\nInstructions:'
    prompt_text += '\n1)Answer each question with `Yes` or `No`. '
    prompt_text += '\n2)Each line of output must be in the format:'
    prompt_text += '\n{Question No.}. Answer: {Answer}'
    
    print(prompt_text)
    print('*' * 100)
    response = llm.query_llm(prompt_text)
    print(response)

def read_input():
    with open('./prompt/example.pmt') as f:
        prompt = f.read()
    return prompt

def example_3():
    f_o_log = open('./prompt/log/llama_log.txt', 'w')
    llm.set_logger(f_o_log)
    prompt = read_input()
    print('prompt llama')
    response = llm.query_llm(prompt)
    print('ok')
    
if __name__ == '__main__':
    example_1()
from question_gen import QuestionGenerator
import json
DATASET = 'pneuma_chicago'

def read_sql_info():
    sql_info_lst = []
    with open(f'./prompt/log/{DATASET}/sql_info.jsonl') as f:
        for line in f:
            sql_info = json.loads(line)
            sql_info_lst.append(sql_info)
    return sql_info_lst

def read_back_response():
    with open(f'./prompt/log/{DATASET}/response.txt') as f:
        response = f.read()
    return response

if __name__ == '__main__':
    obj = QuestionGenerator(DATASET, 'prompt')
    sql_info_lst = read_sql_info()
    response = read_back_response()
    import pdb; pdb.set_trace()
    obj.check_back_sql(sql_info_lst, response)
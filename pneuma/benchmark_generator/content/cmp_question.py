import argparse
import json

def read_data(dataset, tag=None):
    data = []
    if tag is None:
        input_file = f'output/{dataset}/{dataset}_questions_annotated.jsonl'
    else:
        input_file = f'output/{dataset}/{tag}/{dataset}_questions_annotated.jsonl'
    with open(input_file) as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data

def main():
    args = get_args()
    new_q_data = read_data(args.dataset)
    old_q_data = read_data(args.dataset, 'old')

    assert len(new_q_data) == len(old_q_data)
    count = 0
    for offset, new_item in enumerate(new_q_data):
        old_item = old_q_data[offset]
        assert new_item['id'] == old_item['id']
        cond_lst = new_item['meta']['sql_struct']['where']
        eq_cond_lst = [a for a in cond_lst if a['op'] == '=']
        if len(eq_cond_lst) > 1:
            continue
        count += 1
        try:
            assert set(new_item['answer_tables']) == set(old_item['answer_tables'])
        except:
            print('err', 'condtions=', cond_lst, new_item['meta']['sql_struct']['options'])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__  == '__main__':
    main()

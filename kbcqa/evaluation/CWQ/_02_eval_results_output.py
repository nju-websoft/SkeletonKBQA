import json
import random
from evaluation.CWQ._03_eval_script_P1 import proprocess
from common.hand_files import write_json

import random_seed


def get_answers_names(answers_json):
    names_list = []
    if len(answers_json) == 0:
        return names_list
    if len(answers_json) > 1:
        answer_json = random.sample(answers_json, 1)[0]
    else: #1
        answer_json = answers_json[0]

    for ans in answer_json['answer']:
        names_list.append(proprocess(str(ans).lower().strip()))

    if len(names_list) == 0:
        for alias in answer_json['aliases']:
            names_list.append(proprocess(str(alias).lower().strip()))
    return names_list


if __name__ == '__main__':
    prediction_file_path = './sparqa_results/2021.03.04_output_cwq_IR_9_v0.1_wo_agg_withnames_all_nonull.json'
    output_file_path = './sparqa_results/2021.03.04_output_cwq_IR_9_v0.1_wo_agg_withnames_all_nonull_seed9.json'
    # [{"ID": "WebQTest-832_c334509bb5e02cacae1ba2e80c176499", "answer": "2012 world series"},
    sparql_result_list = []
    with open(prediction_file_path) as prediction_file:
        predictions = json.load(prediction_file)
        for index, prediction in enumerate(predictions):  # 遍历prediction
            one_sparql_result = dict()
            one_sparql_result['ID'] = prediction['ID']
            system_answer_names = get_answers_names(prediction['answers'])
            one_sparql_result['answer'] = system_answer_names[0] if len(system_answer_names) > 0 else ""
            sparql_result_list.append(one_sparql_result)

    write_json(sparql_result_list, output_file_path)

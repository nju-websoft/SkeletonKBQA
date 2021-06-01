import json
from common.hand_files import write_set


def get_answers_names(answersdict_list):
    names_list = []
    for ans_dict in answersdict_list:
        if len(ans_dict['answer']) > 0:
            select_answer = ans_dict['answer'][0]
            select_answer = str(select_answer) if isinstance(select_answer, int) else select_answer
            names_list.append(select_answer)
        elif len(ans_dict['aliases']) > 0:
            select_answer = ans_dict['aliases'][0]
            select_answer = str(select_answer) if isinstance(select_answer, int) else select_answer
            names_list.append(select_answer)
    return names_list


m = {}
m['qid'] = 0
m['time'] = 1
m['answers'] = 2
m['predictions'] = 3
m['structure'] = 4
m['function'] = 5
m['answer_cardinality'] = 6
m['commonness'] = 7
m['precision'] = 8
m['recall'] = 9
m['f1'] = 10


res_file = './test_gold.res'
# Go over all lines, record information, and compute recall, precision and F1
res = []
with open(res_file, encoding='utf-8') as f:
  for line in f:
    if len(line) == 0 or line[0] == '#':
      continue
    tokens = line.split('\t')
    qid = int(tokens[m['qid']])
    time = float(tokens[m['time']])
    answers = json.loads(tokens[m['answers']])
    predictions = tokens[m['predictions']] #json.loads(tokens[m['predictions']])
    structure = tokens[m['structure']]
    function = tokens[m['function']]
    answer_cardinality = int(tokens[m['answer_cardinality']])
    commonness = float(tokens[m['commonness']])
    res.append([qid, time, answers, predictions, structure,
                function, answer_cardinality, commonness])


prediction_file_path = './sparqa_results/2021.03.03_output_GraphQ_IR_6_v0.1_wo_agg_E2E_withnames_all_nonull.json'
output_file_path = './sparqa_results/2021.03.03_output_GraphQ_IR_6_v0.1_wo_agg_E2E_withnames_all_nonull_result.res'

sparqa_result_dict = dict()
with open(prediction_file_path) as prediction_file:
    predictions = json.load(prediction_file)
    for index, prediction in enumerate(predictions):  # 遍历prediction
        qid = prediction['ID']
        system_answer_names = get_answers_names(prediction['answers'])
        sparqa_result_dict[qid] = system_answer_names

lines = []
for qid, time, answers, predictions, structure, function, answer_cardinality, commonness in res:
    sparqa_results = []
    if qid in sparqa_result_dict:
        sparqa_results = sparqa_result_dict[qid]
    lines.append('\t'.join([str(qid), '0', json.dumps(answers), json.dumps(sparqa_results),
                     str(structure), str(function), str(answer_cardinality), str(commonness)]))

write_set(lines, output_file_path)

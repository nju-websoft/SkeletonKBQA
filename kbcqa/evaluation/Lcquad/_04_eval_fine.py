

import json
from datasets_interface.question_interface.lcquad_1_0_interface import lcquad_test_list

prediction_file_path = './sparqa_results/2021.02.22_output_LC_SP_1_E2E_withnames_all_nonull【定稿】.json'
qid_to_prf1_dict = dict()
with open(prediction_file_path) as prediction_file:
    predictions = json.load(prediction_file)
    for index, prediction in enumerate(predictions):
        f1 = prediction['f1_score']
        recall = prediction['recall_score']
        precision = prediction['precision_score']
        qid_to_prf1_dict[prediction['ID']] = (precision, recall, f1)

for lcquad_struct in lcquad_test_list:
    question = lcquad_struct.question_normal
    qid = lcquad_struct.qid
    precision, recall, f1 = 0, 0, 0
    if qid in qid_to_prf1_dict:
        precision, recall, f1 = qid_to_prf1_dict[qid]
    print(('%s\t%s\t%.4f\t%.4f\t%.4f') % (qid, question, precision, recall, f1))


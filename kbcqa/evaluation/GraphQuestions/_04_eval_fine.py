import json
from datasets_interface.question_interface.graphquestion_interface import test_graph_questions_struct


prediction_file_path = './sparqa_results/2021.02.22_output_GraphQ_SP_2_v0.2_w_agg_E2E_withnames_all_nonull【定稿】.json'
qid_to_prf1_dict = dict()
with open(prediction_file_path) as prediction_file:
    predictions = json.load(prediction_file)
    for index, prediction in enumerate(predictions):
        f1 = prediction['f1_score']
        recall = prediction['recall_score']
        precision = prediction['precision_score']
        qid_to_prf1_dict[prediction['ID']] = (precision, recall, f1)

for data_ann in test_graph_questions_struct:
    qid = data_ann.qid
    question = data_ann.question
    precision, recall, f1 = 0, 0, 0
    if qid in qid_to_prf1_dict:
        precision, recall, f1 = qid_to_prf1_dict[qid]
    print(('%s\t%s\t%.4f\t%.4f\t%.4f') % (qid, question ,precision, recall, f1))

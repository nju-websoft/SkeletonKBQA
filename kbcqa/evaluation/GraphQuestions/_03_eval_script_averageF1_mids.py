from common.hand_files import read_list_yuanshi

import json

if __name__ == '__main__':
    averageRecall = 0
    averagePrecision = 0
    averageF1 = 0
    count = 0
    prediction_file_path = './sparqa_results/2021.03.02_output_GraphQ_IR_5_v0.1_wo_agg_E2E_withnames_all_nonull.json'
    qid_to_f1_dict = dict()
    with open(prediction_file_path) as prediction_file:
        predictions = json.load(prediction_file)
        for index, prediction in enumerate(predictions):
            f1 = prediction['f1_score']
            recall = prediction['recall_score']
            precision = prediction['precision_score']
            qid_to_f1_dict[str(prediction['ID'])] = f1
            averageRecall += recall
            averagePrecision += precision
            averageF1 += f1
            count += 1

    test_qid_list = read_list_yuanshi('../../../kbcqa_del/evaluation/GraphQuestions/test_qid')
    for test_qid in test_qid_list:
        f1 = 0
        if test_qid in qid_to_f1_dict:
            f1 = qid_to_f1_dict[test_qid]
        print(test_qid, f1)


    count = 2608
    """Print final results"""
    averageRecall = float(averageRecall) / count
    averagePrecision = float(averagePrecision) / count
    averageF1 = float(averageF1) / count
    returnString = ""
    returnString += "Number of questions: " + str(count)
    returnString += "\n"
    returnString += "Average recall over questions: " + str(averageRecall)
    returnString += "\n"
    returnString += "Average precision over questions: " + str(averagePrecision)
    returnString += "\n"
    returnString += "Average f1 over questions (accuracy): " + str(averageF1)
    returnString += "\n"
    if averagePrecision + averageRecall > 0:
        averageNewF1 = 2 * averageRecall * averagePrecision / (averagePrecision + averageRecall)
    else:
        averageNewF1 = 0.0
    returnString += "F1 of average recall and average precision: " + str(averageNewF1)
    print(returnString)



from evaluation.Lcquad._03_eval_script_QALDF1_fromQALD import calculate, calculate_f1_qald
from datasets_interface.question_interface import lcquad_1_0_interface
from common.hand_files import read_list_yuanshi


import json
if __name__ == '__main__':
    ps, rs = [], []

    """==files=="""
    prediction_file_path = './sparqa_results/2021.02.22_output_LC_SP_1_E2E_withnames_all_nonull【定稿】.json'
    """"""
    questions = read_list_yuanshi('./test_tworelation_questions.txt')
    """"""

    question_to_type_dict = dict()
    question_to_denotation_dict = dict()
    with open(prediction_file_path) as prediction_file:
        predictions = json.load(prediction_file)
        for index, prediction in enumerate(predictions):  # 遍历prediction
            question_to_type_dict[prediction['question_normal']] = prediction['question_type']
            question_to_denotation_dict[prediction['question_normal']] = prediction['answers_id']

    counts = []
    for i, lcquad_struct in enumerate(lcquad_1_0_interface.lcquad_test_list):
        question_normal = lcquad_struct.question_normal
        if question_normal not in questions:
            continue

        gold_question_type = lcquad_1_0_interface.get_type_by_question(question=question_normal)
        gold_answers = lcquad_1_0_interface.get_answers_by_question(question=question_normal)

        predict_question_type, predict_denotation = None, None
        if question_normal in question_to_type_dict:
            predict_question_type = question_to_type_dict[question_normal]
        if question_normal in question_to_denotation_dict:
            predict_denotation = question_to_denotation_dict[question_normal]

        p, r = 0, 0
        if gold_question_type != predict_question_type:
            count = calculate(goldList=[1], predictedList=[0])
        else:
            if gold_question_type == 'ask': #ASK
                if gold_answers == predict_denotation:
                    count = calculate(goldList=[1], predictedList=[1])
                else:
                    count = calculate(goldList=[1], predictedList=[0])
            else: # COUNT
                if gold_answers == 'count':
                    if gold_answers == predict_denotation:
                        count = calculate(goldList=[1], predictedList=[1])
                    else:
                        count = calculate(goldList=[1], predictedList=[0])
                else: # SELECT
                    count = calculate(goldList=gold_answers, predictedList=predict_denotation)
        counts.append(count)

    macro_precision, macro_recall, macro_f1, qald_f1 = calculate_f1_qald(counts=counts)
    print('#counts length:\t', len(counts))
    print ('#macro_precision:\t',macro_precision)
    print ('#macro_recall:\t',macro_recall)
    print ('#macro_f1:\t',macro_f1)
    print ('#qald_f1:\t',qald_f1)

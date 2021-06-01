import numpy as np

from datasets_interface.question_interface import lcquad_1_0_interface


def p_r_every_q(question, gold_question_type, gold_answers, predict_question_type, predict_denotation):
    '''
    :param gold_question_type: ask, count, bgp
    :param gold_answers:  ask [1],  count [9],  bgp ['http://dbpedia.org/resource/Ella_Fitzgerald']
    :param predict_function: None, count,
    :param predict_compositionality_type:  conj, nest, ask, simple,
    :param predict_denotation:  ['http://dbpedia.org/resource/Ella_Fitzgerald']
    :return:
    '''

    p, r = 0, 0
    if gold_question_type != predict_question_type:
        p, r = 0, 0
    else:
        if gold_question_type == 'ask':
            if gold_answers == predict_denotation:
                p, r = 1, 1
            else:
                p, r = 0, 0
        else:
            # COUNT
            if gold_answers == 'count':
                if gold_answers == predict_denotation:
                    p, r = 1,1
                else:
                    p, r = 0,0
            #SELECT
            else:
                n_gs_answers = len(gold_answers)
                n_answers = len(predict_denotation)
                n_correct = len(set(gold_answers) & set(predict_denotation))
                try:
                    r = float(n_correct) / n_gs_answers
                except ZeroDivisionError:
                    print(question)

                try:
                    p = float(n_correct) / n_answers
                except ZeroDivisionError:
                    print(question)
    return p,r


import json
if __name__ == '__main__':
    ps, rs = [], []

    prediction_file_path = './sparqa_results/2021.02.22_output_LC_SP_1_E2E_withnames_all_nonull.json'
    question_to_type_dict = dict()
    question_to_denotation_dict = dict()
    with open(prediction_file_path) as prediction_file:
        predictions = json.load(prediction_file)
        for index, prediction in enumerate(predictions):  # 遍历prediction
            question_to_type_dict[prediction['question_normal']] = prediction['question_type']
            question_to_denotation_dict[prediction['question_normal']] = prediction['answers_id']

    for i, lcquad_struct in enumerate(lcquad_1_0_interface.lcquad_test_list):
        question_normal = lcquad_struct.question_normal
        gold_question_type = lcquad_1_0_interface.get_type_by_question(question=question_normal)
        gold_answers = lcquad_1_0_interface.get_answers_by_question(question=question_normal)
        predict_question_type, predict_denotation = None, None
        if question_normal in question_to_type_dict:
            predict_question_type = question_to_type_dict[question_normal]
        if question_normal in question_to_denotation_dict:
            predict_denotation = question_to_denotation_dict[question_normal]
        p, r = p_r_every_q(question=question_normal,
                           gold_question_type=gold_question_type, gold_answers=gold_answers,
                           predict_question_type=predict_question_type, predict_denotation=predict_denotation)
        f1 = 0
        if p + r > 0:
            f1 = (2*p*r)/(p+r)
        ps.append(p)
        rs.append(r)
        print(('%s\t%s\t%s\t%.4f\t%.4f\t%.4f') % (lcquad_struct.qid, question_normal, gold_question_type, p, r, f1))

    precision = np.mean(ps)
    recall = np.mean(rs)
    print(len(ps), len(rs))
    print("P: %.10f R: %.10f"%(np.sum(ps), np.sum(recall)))
    print("P: %.10f R: %.10f"%(precision, recall))
    f1 = 0
    if precision + recall > 0:
        f1 = 2 * recall * precision / (precision + recall)
    print(f1)


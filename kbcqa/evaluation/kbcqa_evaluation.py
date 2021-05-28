import os
import collections
from scipy.special import softmax
import numpy as np

from evaluation import sempre_evaluation
from common.hand_files import read_structure_file, write_structure_file, write_json

lcquad_ask_thresold = 0.5


def run_end_to_end_evaluation(structure_with_2_2_grounded_graph_folder, dataset='cwq',
                              output_file='./2020.01.21_output_cwq_IR5_withnames_all_nonull_comparative.json'):
    if dataset == 'lcquad':
        get_denotations_by_score_standard_binglie(structure_with_2_2_grounded_graph_folder, output_file=output_file)
    elif dataset == 'cwq':
        get_denotations_by_score_standard_prediction(input_file=structure_with_2_2_grounded_graph_folder, dataset=dataset, output_file=output_file)
    elif dataset == 'graphq':
        get_denotations_by_score_standard_prediction(input_file=structure_with_2_2_grounded_graph_folder, dataset=dataset, output_file=output_file)


def computed_every_grounded_graph_f1_lcquad(input_file):
    from datasets_interface.question_interface import lcquad_1_0_interface
    for structure_path in os.listdir(input_file):
        structure_with_grounded_graphq_file = input_file + structure_path
        print(structure_path)
        structure_list = read_structure_file(structure_with_grounded_graphq_file)
        for structure in structure_list:
            gold_answer_mid_set = lcquad_1_0_interface.get_answers_by_question(structure.question) #['http://dbpedia.org/resource/Colorado']
            for ungrounded_graph in structure.ungrounded_graph_forest:
                for grounded_graph in ungrounded_graph.get_grounded_graph_forest():
                    system_denotation_set = set(grounded_graph.denotation)
                    recall, precision, f1 = sempre_evaluation.computeF1(gold_answer_mid_set, system_denotation_set)
                    grounded_graph.f1_score = f1
                    grounded_graph.recall_score = recall
                    grounded_graph.precision_score = precision
        write_structure_file(structure_list, input_file + structure_path)


def computed_every_grounded_graph_f1_cwq(input_file):
    from datasets_interface.question_interface import complexwebquestion_interface
    all_structure_path = os.listdir(input_file)
    error_list = []
    for structure_path in all_structure_path:
        structure_with_grounded_graphq_file = input_file + structure_path
        print(structure_path)
        try:
            structure_list = read_structure_file(structure_with_grounded_graphq_file)
            for structure in structure_list:
                gold_answer_mid_set = complexwebquestion_interface.get_answers_by_question(structure.question)
                for ungrounded_graph in structure.ungrounded_graph_forest:
                    for grounded_graph in ungrounded_graph.get_grounded_graph_forest():
                        system_denotation_set = set(grounded_graph.denotation)
                        recall, precision, f1 = sempre_evaluation.computeF1(gold_answer_mid_set, system_denotation_set)
                        grounded_graph.f1_score = f1
                        grounded_graph.recall_score = recall
                        grounded_graph.precision_score = precision
            write_structure_file(structure_list, input_file + structure_path)
        except Exception as e:
            print('error')
            error_list.append(structure_path)
    print('error_list:\t', error_list)


def computed_every_grounded_graph_f1_graphq(input_file):
    from datasets_interface.question_interface import graphquestion_interface
    for structure_path in os.listdir(input_file):
        structure_with_grounded_graphq_file = input_file + structure_path
        print(structure_path)
        structure_list = read_structure_file(structure_with_grounded_graphq_file)
        for structure in structure_list:
            gold_answers_mid_set = graphquestion_interface.get_answers_mid_by_question(structure.question)
            for ungrounded_graph in structure.ungrounded_graph_forest:
                for grounded_graph in ungrounded_graph.get_grounded_graph_forest():
                    new_system_answers_list = []
                    for system_answer in set(grounded_graph.denotation):
                        if isinstance(system_answer, int):
                            new_system_answers_list.append(str(system_answer))
                        else:
                            new_system_answers_list.append(system_answer)
                    recall, precision, f1 = sempre_evaluation.computeF1(gold_answers_mid_set, new_system_answers_list)
                    grounded_graph.f1_score = f1
                    grounded_graph.recall_score = recall
                    grounded_graph.precision_score = precision
                    if f1 > 0:
                        print(structure_path, f1) # print(structure_path, gold_answers_mid_set, new_system_answers_list, f1)
            structure.gold_answer = gold_answers_mid_set # update answers by answer mid list   ["Kimberly-Clark"]  ['en.kimberly-clark']
        write_structure_file(structure_list, input_file + structure_path)


def compute_all_questions_recall(input_file):
    '''
    # oracle all recall by max f1
    :param input_file:
    :return:
    '''
    all_data_path = os.listdir(input_file)
    all_recall = 0
    error_list = []
    for path in all_data_path:
        try:
            structure_list = read_structure_file(input_file + path)
            max_f1 = 0
            question = None
            for structure in structure_list:
                question = structure.question
                for ungrounded_graph in structure.ungrounded_graph_forest:
                    for grounded_graph in ungrounded_graph.get_grounded_graph_forest():
                        if max_f1 < grounded_graph.f1_score:
                            max_f1 = grounded_graph.f1_score
            all_recall += max_f1
            if max_f1 != 1.0:
                print(('%s\t%s\t%s') % (path, question, str(max_f1)))
        except Exception as e:
            print(e)
            error_list.append(path)
    print('#error_list:\t', error_list)
    print(all_recall, len(all_data_path))


def get_denotations_by_score_standard_binglie(input_file, output_file='./e2e_2021.01.20_lcquad_predict_IR5_update.json'):
    prediction_list = []
    for structure_path in os.listdir(input_file):
        question_normal = None
        question_type = None
        question_qid = None
        print(structure_path)
        totalscore_queryid_sparql = collections.defaultdict(list)
        grounded_query_id_denotation = collections.defaultdict(set)
        grounded_query_id_predictscore = collections.OrderedDict()
        grounded_query_id_f1 = collections.defaultdict(set)
        grounded_query_id_to_recall = collections.defaultdict(set)
        grounded_query_id_to_precision = collections.defaultdict(set)
        grounded_query_id_keypath = collections.defaultdict()
        for structure in read_structure_file(input_file + structure_path):
            question_normal = structure.question
            question_qid = structure.qid
            question_type = structure.compositionality_type

            for j, ungrounded_graph in enumerate(structure.ungrounded_graph_forest):
                if j != len(structure.ungrounded_graph_forest)-1:
                    continue
                for grounded_graph in ungrounded_graph.get_grounded_graph_forest():
                    totalscore_queryid_sparql[grounded_graph.combine_score].append(grounded_graph.grounded_query_id) #score  total_score  combine_score
                    grounded_query_id_denotation[grounded_graph.grounded_query_id] = grounded_graph.denotation
                    grounded_query_id_predictscore[grounded_graph.grounded_query_id] = grounded_graph.score
                    grounded_query_id_f1[grounded_graph.grounded_query_id] = grounded_graph.f1_score
                    grounded_query_id_to_recall[grounded_graph.grounded_query_id] = grounded_graph.recall_score
                    grounded_query_id_to_precision[grounded_graph.grounded_query_id] = grounded_graph.precision_score
                    grounded_query_id_keypath[grounded_graph.grounded_query_id] = grounded_graph.key_path

        if question_type == 'ask':
            predict_denotation = [0]
            predict_score_list = []
            for _, temp_score in grounded_query_id_predictscore.items():
                predict_score_list.append(temp_score)
            score_list = softmax(np.array(predict_score_list))  # np.array([-6.8602, -6.8602, -8.3860, -7.4321])
            max_index = score_list.argmax()
            # print(score_list, max_score, keypath_list[max_index])
            # #[2.55135490e-05 2.49768139e-05 2.20922475e-05 7.50664797e-06, 4.22800427e-04 9.85553435e-01 1.27269936e-02] 0.9855534350037205
            if score_list[max_index] >= lcquad_ask_thresold:
                predict_denotation = [1]
            q_dict = collections.OrderedDict()
            q_dict['ID'] = question_qid
            q_dict['question_normal'] = question_normal
            q_dict['question_type'] = question_type
            q_dict['answers_id'] = predict_denotation
            q_dict['answers'] = []
            q_dict['f1_score'] = predict_denotation[0]
            q_dict['recall_score'] = predict_denotation[0]
            q_dict['precision_score'] = predict_denotation[0]
            prediction_list.append(q_dict)

        else: #bgp, count
            totalscore_queryid_sparql = dict(sorted(totalscore_queryid_sparql.items(), key=lambda d: d[0], reverse=True))
            for totalscore, grounded_query_ids in totalscore_queryid_sparql.items():
                # 从 并列中选择一个 有并列, 选dbo
                system_grounded_query_id = None
                system_new_key_path_ = None
                system_predicates = None
                for grounded_query_id in grounded_query_ids:
                    predicates = grounded_query_id_keypath[grounded_query_id].split('\t')
                    new_key_path_ = []
                    for predicate in predicates:
                        new_key_path_.append(predicate.split('/')[-1])
                    if system_grounded_query_id is None:
                        system_grounded_query_id = grounded_query_id
                        system_new_key_path_ = new_key_path_
                        system_predicates = predicates
                    elif system_new_key_path_ == '\t'.join(new_key_path_): #谁是dbo, 就选谁
                        is_system_all_dbo = True
                        for system_predicate in system_predicates:
                            if 'http://dbpedia.org/ontology/' not in system_predicate:
                                is_system_all_dbo = False
                        is_current_all_dbo = True
                        for current_predicate in predicates:
                            if 'http://dbpedia.org/ontology/' not in current_predicate:
                                is_current_all_dbo = False
                        if is_system_all_dbo:
                            break
                        elif is_current_all_dbo:
                            system_grounded_query_id = grounded_query_id
                            break
                q_dict = collections.OrderedDict()
                q_dict['ID'] = question_qid
                q_dict['question_normal'] = question_normal
                q_dict['question_type'] = question_type
                q_dict['answers_id'] = grounded_query_id_denotation[system_grounded_query_id]
                q_dict['answers'] = []
                q_dict['f1_score'] = grounded_query_id_f1[system_grounded_query_id]
                q_dict['recall_score'] = grounded_query_id_to_recall[system_grounded_query_id]
                q_dict['precision_score'] = grounded_query_id_to_precision[system_grounded_query_id]
                prediction_list.append(q_dict)
                break

    write_json(prediction_list, pathfile=output_file)


def get_denotations_by_score_standard_prediction(input_file, dataset='cwq', output_file = './2020.01.21_output_cwq_IR5_withnames_all_nonull_comparative.json'):
    from common.hand_files import write_json
    assert dataset in ['cwq', 'graphq']
    if dataset == 'cwq':
        from evaluation.CWQ import _01_mid_to_label_alias_names
    elif dataset == 'graphq':
        from evaluation.GraphQuestions import _01_mid_to_label_alias_names
    prediction_list = []
    for structure_path in os.listdir(input_file):
        print(structure_path)
        structure_list = read_structure_file(input_file + structure_path)
        score_to_queryid_sparql = collections.defaultdict(list)
        grounded_query_id_to_f1 = collections.defaultdict(set)
        grounded_query_id_to_recall = collections.defaultdict(set)
        grounded_query_id_to_precision = collections.defaultdict(set)
        grounded_query_id_to_denotation = collections.defaultdict(set)
        qid = None
        for structure in structure_list:
            qid = structure.qid
            for j, ungrounded_graph in enumerate(structure.ungrounded_graph_forest):
                if j != len(structure.ungrounded_graph_forest) - 1:
                    continue
                for grounded_graph in ungrounded_graph.get_grounded_graph_forest():
                    score_to_queryid_sparql[grounded_graph.score].append(grounded_graph.grounded_query_id) #score total_score, combine_score
                    grounded_query_id_to_denotation[grounded_graph.grounded_query_id] = grounded_graph.denotation
                    grounded_query_id_to_f1[grounded_graph.grounded_query_id] = grounded_graph.f1_score
                    grounded_query_id_to_recall[grounded_graph.grounded_query_id] = grounded_graph.recall_score
                    grounded_query_id_to_precision[grounded_graph.grounded_query_id] = grounded_graph.precision_score
        answers_ids = []
        answers = [] # "[{ "answer_id": "m.034tl", "answer": ["Guam"],"aliases": []},]"
        f1_score,recall_score, precision_score = 0,0,0
        #第一个name or alias 非空的, 就跳出来
        is_name_null = True
        score_to_queryid_sparql = dict(sorted(score_to_queryid_sparql.items(), key=lambda d: d[0], reverse=True))
        for totalscore, grounded_query_ids in score_to_queryid_sparql.items():
            for grounded_query_id in grounded_query_ids:
                answers_ids = grounded_query_id_to_denotation[grounded_query_id]
                f1_score = grounded_query_id_to_f1[grounded_query_id]
                recall_score = grounded_query_id_to_recall[grounded_query_id]
                precision_score = grounded_query_id_to_precision[grounded_query_id]
                answers = []
                for answer_id in answers_ids:
                    names_dict = _01_mid_to_label_alias_names.get_names(answer_id)
                    if len(names_dict['answer'])>0 or len(names_dict['aliases'])>0:
                        is_name_null = False
                        answers.append(names_dict)
                if not is_name_null:
                    break
            if not is_name_null:
                break
        q_dict = dict()
        q_dict['ID'] = qid
        q_dict['answers_id'] = answers_ids
        q_dict['answers'] = answers
        q_dict['f1_score'] = f1_score
        q_dict['recall_score'] = recall_score
        q_dict['precision_score'] = precision_score
        prediction_list.append(q_dict)
    _01_mid_to_label_alias_names.write_cache_json()
    write_json(prediction_list, output_file)


import json
from common import globals_args
from common.hand_files import read_json
from datasets_interface import questions_utils


class ComplexWebQuestion:
    '''graphquestion class'''
    def __init__(self):
        self.ID = ''  # int  251000000,
        self.webqsp_ID = ''  # "WebQTrn-3252"
        self.webqsp_question = '' #webqsp_question
        self.machine_question = '' #machine_question
        self.question = '' #question
        self.sparql = '' #sparql
        self.parsed_sparql = ''
        self.compositionality_type = '' #compositionality_type
        self.answers = ''
        self.created = '' # created time


def get_answers_by_id(qid):
    answers = "none"
    for complexwebq_struct in all_complexwebq_list:
        if complexwebq_struct.ID == qid:
            answers = complexwebq_struct.answers
            break
    return answers


def read_complexwebq_question_json(filename):
    '''
    function: read complexquestion dataset
    :param filename: filename path file
    :return: graph_question structure list
    '''
    complexwebq_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        complexwebq_json_data = json.load(f)
    for complexwebq in complexwebq_json_data:
        complexwebq_struct = ComplexWebQuestion()
        complexwebq_struct.ID = complexwebq['ID']
        complexwebq_struct.webqsp_ID = complexwebq['webqsp_ID']
        complexwebq_struct.webqsp_question = complexwebq['webqsp_question']
        complexwebq_struct.machine_question = complexwebq['machine_question']
        if 'question_normal' in complexwebq:
            complexwebq_struct.question = complexwebq['question_normal'][0]
        else:
            complexwebq_struct.question = complexwebq['question']
        complexwebq_struct.sparql = complexwebq['sparql']
        if 'parsed_sparql' in complexwebq:
            complexwebq_struct.parsed_sparql = complexwebq['parsed_sparql']
        complexwebq_struct.compositionality_type = complexwebq['compositionality_type']
        if 'answers' in complexwebq:
            complexwebq_struct.answers = complexwebq['answers']
        else:
            complexwebq_struct.answers = get_answers_by_id(qid=complexwebq_struct.ID)
        complexwebq_struct.created = complexwebq['created']
        complexwebq_list.append(complexwebq_struct)
    return complexwebq_list


all_complexwebq_list = read_complexwebq_question_json(globals_args.fn_cwq_file.complexwebquestion_all_questions_dir)
complexwebq_test_list = read_complexwebq_question_json(globals_args.fn_cwq_file.complexwebquestion_test_dir)
complexwebq_dev_list = read_complexwebq_question_json(globals_args.fn_cwq_file.complexwebquestion_dev_dir)
complexwebq_train_list = read_complexwebq_question_json(globals_args.fn_cwq_file.complexwebquestion_train_dir)

bgp_test_qid_to_graphs_dict = questions_utils.extract_grounded_graph_from_jena_freebase(globals_args.fn_cwq_file.complexwebquestion_test_bgp_dir)
bgp_dev_qid_to_graphs_dict = questions_utils.extract_grounded_graph_from_jena_freebase(globals_args.fn_cwq_file.complexwebquestion_dev_bgp_dir)
bgp_train_qid_to_graphs_dict = questions_utils.extract_grounded_graph_from_jena_freebase(globals_args.fn_cwq_file.complexwebquestion_train_bgp_dir)
annotation_node_questions_json = read_json(globals_args.fn_cwq_file.complexwebquestion_node_ann_dir)


def get_answers_by_question(question_normal=None):
    '''
    "answers": [
            {
                "answer": "Super Bowl XLVII",
                "answer_id": "m.0642vqv",
                "aliases": [
                    "Super Bowl 2013",
                    "Super Bowl 47"
                ]
            }
        ],
    :param question_normal:
    :return:
    '''

    answers = []
    for data_ann in complexwebq_test_list:
        if data_ann.question == question_normal:
            answers = data_ann.answers
            break
    if len(answers) == 0:
        for data_ann in complexwebq_dev_list:
            if data_ann.question == question_normal:
                answers = data_ann.answers
                break
    if len(answers) == 0:
        for data_ann in complexwebq_train_list:
            if data_ann.question == question_normal:
                answers = data_ann.answers
                break
    '''get gold answers from structure'''
    gold_answer_mid_set = set()
    for gold_answer_dict in answers:
        gold_answer_mid_set.add(gold_answer_dict['answer_id'])
    return gold_answer_mid_set


def get_type_by_question(question_normal=None):
    compositionality_type = None
    for data_ann in complexwebq_test_list:
        if data_ann.question == question_normal:
            compositionality_type = data_ann.compositionality_type
            break
    for data_ann in complexwebq_dev_list:
        if data_ann.question == question_normal:
            compositionality_type = data_ann.compositionality_type
            break
    for data_ann in complexwebq_train_list:
        if data_ann.question == question_normal:
            compositionality_type = data_ann.compositionality_type
            break
    return compositionality_type


def get_nodes_by_question(question_normal=None):
    nodes = []
    for data_ann in annotation_node_questions_json:
        if data_ann['question_normal'] == question_normal:
            nodes = data_ann['node_mention_nju'] #node_mention
            break
    return nodes


def get_topic_entities_by_question(question_normal=None):
    entity_list = []
    for data_ann in annotation_node_questions_json:
        if data_ann['question_normal'] == question_normal:
            for entity_dict in data_ann['node_mention_nju']:  #node_mention
                if entity_dict['tag'] == 'entity':
                    entity_list.append(entity_dict['uri'])
            break
    return entity_list


def get_abstract_question_by_question(question=None):
    abstract_question = question
    for data_ann in annotation_node_questions_json:
        if data_ann['question_normal'] == question and 'abstract_question' in data_ann:
            abstract_question = data_ann['abstract_question']
            break
    return abstract_question


def get_topic_literals_by_question(question_normal=None):
    literals_list = []
    for data_ann in annotation_node_questions_json:
        if data_ann['question_normal'] == question_normal:
            for entity_dict in data_ann['node_mention_nju']: #node_mention
                if entity_dict['tag'] == 'literal':
                    literals_list.append(entity_dict['uri'])
            break
    return literals_list


def get_topic_entities_by_question_and_mention(question_normal=None, mention=None):
    result = dict()
    for data_ann in annotation_node_questions_json:
        if data_ann['question_normal'] == question_normal:
            for entity_dict in data_ann['node_mention_nju']: #node_mention
                if entity_dict['mention'] == mention:
                    result[entity_dict['uri']] = 1.0
                    break
            break
    return result


def get_grounded_graph_by_question(qid=None, data=None):
    result = None
    if data == 'test' and qid in bgp_test_qid_to_graphs_dict:
        result = bgp_test_qid_to_graphs_dict[qid]
    elif data == 'dev' and qid in bgp_dev_qid_to_graphs_dict:
        result = bgp_dev_qid_to_graphs_dict[qid]
    elif data == 'train' and qid in bgp_train_qid_to_graphs_dict:
        result = bgp_train_qid_to_graphs_dict[qid]
    return result


def get_sparql_by_question(question_normal=None):
    sparql = None
    for cwq_struct in complexwebq_test_list:
        if question_normal == cwq_struct.question:
            sparql = cwq_struct.sparql
            break
    for cwq_struct in complexwebq_dev_list:
        if question_normal == cwq_struct.question:
            sparql = cwq_struct.sparql
            break
    for cwq_struct in complexwebq_train_list:
        if question_normal == cwq_struct.question:
            sparql = cwq_struct.sparql
            break
    return sparql


def get_q_aggregation_type_by_question(question_normal=None):
    compositionality_type = None
    for data_ann in complexwebq_test_list:
        if data_ann.question == question_normal:
            compositionality_type = data_ann.compositionality_type
            break
    for data_ann in complexwebq_dev_list:
        if data_ann.question == question_normal:
            compositionality_type = data_ann.compositionality_type
            break
    for data_ann in complexwebq_train_list:
        if data_ann.question == question_normal:
            compositionality_type = data_ann.compositionality_type
            break
    return compositionality_type


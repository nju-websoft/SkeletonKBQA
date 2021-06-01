import json
from common.globals_args import fn_lcquad_file
from common.hand_files import read_json
from datasets_interface import questions_utils


class LCQuAD_question:

    '''LCQuAD_question class'''
    def __init__(self, qid=None, question=None, question_normal=None, verbalized_question=None,sparql_template_id=None):
        self.qid = qid
        self.question = question
        self.question_normal = question_normal
        self.sparql = ''
        self.parsed_sparql = {}
        self.verbalized_question = verbalized_question
        self.sparql_template_id = sparql_template_id

    def __str__(self):
        print_str = '{'
        print_str += 'qid:' + self.qid
        print_str += ',\tquestion_normal:' + self.question_normal
        print_str += '}'
        return print_str


def read_train_test_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    lcquad_list = []
    for question_ann in data:
        lcquad_annotation = LCQuAD_question()
        lcquad_annotation.question_normal = question_ann['normal_question']
        lcquad_annotation.question = question_ann['corrected_question']
        lcquad_annotation.sparql = question_ann['sparql_query']
        lcquad_annotation.parsed_sparql = question_ann['parsed_sparql']
        lcquad_annotation.sparql_template_id = question_ann['sparql_template_id']
        lcquad_annotation.verbalized_question = question_ann['intermediary_question']
        lcquad_annotation.qid = question_ann['_id']
        lcquad_list.append(lcquad_annotation)
    return lcquad_list


lcquad_test_list = read_train_test_data(filepath=fn_lcquad_file.lcquad_test_dir)
lcquad_train_list = read_train_test_data(filepath=fn_lcquad_file.lcquad_train_dir)
bgp_test_qid_to_graphs_dict = questions_utils.extract_grounded_graph_from_jena_dbpedia(fn_lcquad_file.lcquad_test_bgp_dir)
bgp_train_qid_to_graphs_dict = questions_utils.extract_grounded_graph_from_jena_dbpedia(fn_lcquad_file.lcquad_train_bgp_dir)
annotation_node_answers_all_questions_json = read_json(fn_lcquad_file.lcquad_all_q_node_ann_dir)


def get_answers_by_question(question=None):
    answers = []
    for data_ann in annotation_node_answers_all_questions_json:
        if data_ann['question_normal'] == question:
            answers = data_ann['answers']
    return answers


def get_type_by_question(question=None):
    question_type = 'bgp'
    for data_ann in annotation_node_answers_all_questions_json:
        if data_ann['question_normal'] == question:
            question_type = data_ann['type']
    return question_type


def get_type_by_qid(qid=None):
    question_type = []
    for data_ann in annotation_node_answers_all_questions_json:
        if data_ann['SerialNumber'] == qid:
            question_type = data_ann['type']
    return question_type


def get_nodes_by_question(question=None):
    nodes = []
    for data_ann in annotation_node_answers_all_questions_json:
        if data_ann['question_normal'] == question:
            nodes = data_ann['node_mention']
            break
    return nodes


def get_abstract_question_by_question(question=None):
    abstract_question = question
    for data_ann in annotation_node_answers_all_questions_json:
        if data_ann['question_normal'] == question and 'abstract_question' in data_ann:
            abstract_question = data_ann['abstract_question']
    return abstract_question


def get_topic_entities_by_question(question=None):
    entity_list = []
    for data_ann in annotation_node_answers_all_questions_json:
        if data_ann['question_normal'] == question and 'node_mention' in data_ann:
            for entity_dict in data_ann['node_mention']:
                if entity_dict['tag'] == 'entity':
                    entity_list.append(entity_dict['uri'])
            break
        elif data_ann['question'] == question and 'node_mention' in data_ann:
            for entity_dict in data_ann['node_mention']:
                if entity_dict['tag'] == 'entity':
                    entity_list.append(entity_dict['uri'])
            break
    return entity_list


def get_parsed_sparql_by_question(question=None):
    parsed_sparql = None
    for lcquad_struct in lcquad_test_list:
        if question == lcquad_struct.question_normal:
            parsed_sparql = lcquad_struct.parsed_sparql
    for lcquad_struct in lcquad_train_list:
        if question == lcquad_struct.question_normal:
            parsed_sparql = lcquad_struct.parsed_sparql
    return parsed_sparql


def get_topic_entities_by_question_and_mention(question=None, mention=None):
    result = dict()
    for data_ann in annotation_node_answers_all_questions_json:
        if data_ann['question_normal'] == question:
            for entity_dict in data_ann['node_mention']:
                if entity_dict['mention'] == mention:
                    result[entity_dict['uri']] = 1.0
                    break
    return result


def get_grounded_graph_by_question(qid=None, data=None):
    result = None
    if data == 'test' and qid in bgp_test_qid_to_graphs_dict:
        result = bgp_test_qid_to_graphs_dict[qid]
    elif data == 'train' and qid in bgp_train_qid_to_graphs_dict:
        result = bgp_train_qid_to_graphs_dict[qid]
    return result


def get_sparql_by_question(question=None):
    sparql = None
    for lcquad_struct in lcquad_test_list:
        if question == lcquad_struct.question_normal:
            sparql = lcquad_struct.sparql
    for lcquad_struct in lcquad_train_list:
        if question == lcquad_struct.question_normal:
            sparql = lcquad_struct.sparql
    return sparql


def get_q_aggregation_type_by_question(question=None):
    pass


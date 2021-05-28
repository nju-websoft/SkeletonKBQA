import json
from common_structs.grounded_graph import GroundedNode, GroundedEdge, GrounedGraph
from common.hand_files import read_json
from common import globals_args


class GraphQuestion:
    '''graphquestion class'''
    def __init__(self):
        self.qid = ''  # int  251000000,
        self.question = ''  # string  "xtracycle is which type of bicycle?",
        self.answer = ''  # list  ["Longtail"]
        self.answer_mid = []
        self.function = ''  # string  "none"
        self.commonness = ''  # float   -19.635822428214723,
        self.num_node = ''  # int  2
        self.num_edge = ''  # int  1
        self.graph_query = ''  # dict
        self.nodes = []  # list
        self.edges = []  # list
        self.sparql_query = ''  # string   PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> \nSELECT (?x0 AS ?value) WHERE {\nSELECT DISTINCT ?x0  WHERE { \n?x0 :type.object.type :bicycles.bicycle_type . \nVALUES ?x1 { :en.xtracycle } \n?x1 :bicycles.bicycle_model.bicycle_type ?x0 . \nFILTER ( ?x0 != ?x1  )\n}\n}
        self.parsed_sparql = ''


def read_graph_question_json(filename):
    '''
    function: read graphquestion dataset
    :param filename: filename path file
    :return: graph_question structure list
    '''
    graphquestionsList = list()
    with open(filename, 'rb') as f:
        data = json.load(f)
    for questionAnnotation in data:
        graphq = GraphQuestion()
        graphq.qid = questionAnnotation["qid"]
        graphq.graph_entity_level_paraphrase_id = graphq.qid % 100
        graphq.graph_sentence_level_paraphrase_id = (graphq.qid // 100) % 10000
        graphq.graph_query_id = graphq.qid // 1000000
        # graphq.question = questionAnnotation["question"]
        graphq.question = questionAnnotation["question_normal"]
        graphq.answer = questionAnnotation["answer"]
        graphq.answer_mid = questionAnnotation["answer_mid"]
        graphq.function = questionAnnotation["function"]
        graphq.commonness = questionAnnotation["commonness"]
        graphq.num_node = questionAnnotation["num_node"]
        graphq.num_edge = questionAnnotation["num_edge"]
        graphq.graph_query = questionAnnotation["graph_query"]
        for node in questionAnnotation["graph_query"]["nodes"]:
            graphq.nodes.append(GroundedNode(nid=node["nid"], node_type=node["node_type"], type_class=node["class"],
                friendly_name=node["friendly_name"], question_node=node["question_node"], function=node["function"], id=node["id"], score=1.0))
        for edge in questionAnnotation["graph_query"]["edges"]:
            graphq.edges.append(GroundedEdge(start=edge["start"], end=edge["end"], relation=edge["relation"], friendly_name=edge["friendly_name"], score=1.0))
        graphq.sparql_query = questionAnnotation["sparql_query"]
        graphq.parsed_sparql = questionAnnotation['parsed_sparql']
        graphquestionsList.append(graphq)
    return graphquestionsList


test_graph_questions_struct = read_graph_question_json(globals_args.fn_graph_file.graphquestions_testing_dir)
train_graph_questions_struct = read_graph_question_json(globals_args.fn_graph_file.graphquestions_training_dir)
annotation_node_questions_json = read_json(globals_args.fn_graph_file.graphquestions_node_ann_dir)


def get_answers_by_question(question=None):
    answers = []
    for data_ann in test_graph_questions_struct:
        if data_ann.question == question:
            answers = data_ann.answer
            break
    for data_ann in train_graph_questions_struct:
        if data_ann.question == question:
            answers = data_ann.answer
    return answers


def get_answers_mid_by_question(question=None):
    answers = []
    for data_ann in test_graph_questions_struct:
        if data_ann.question == question:
            answers = data_ann.answer_mid
            break
    for data_ann in train_graph_questions_struct:
        if data_ann.question == question:
            answers = data_ann.answer_mid
    # [80] -> ['80']
    new_gold_answers_set = set()
    for gold_answer in answers:
        if isinstance(gold_answer, int):
            new_gold_answers_set.add(str(gold_answer))
        else:
            new_gold_answers_set.add(gold_answer)
    return list(new_gold_answers_set)


def get_type_by_question(question=None):
    type_ = "none"
    for data_ann in test_graph_questions_struct:
        if data_ann.question == question:
            type_ = data_ann.function
            break
    for data_ann in train_graph_questions_struct:
        if data_ann.question == question:
            type_ = data_ann.function
    return type_


def get_nodes_by_question(question=None):
    nodes = []
    for data_ann in annotation_node_questions_json:
        if data_ann['question_normal'] == question:
            nodes = data_ann['node_mention_nju']
            break
    return nodes


def get_abstract_question_by_question(question=None):
    abstract_question = question
    for data_ann in annotation_node_questions_json:
        if data_ann['question_normal'] == question and 'abstract_question' in data_ann:
            abstract_question = data_ann['abstract_question']
    return abstract_question


def get_topic_entities_by_question(question=None):
    entity_list = []
    for data_ann in annotation_node_questions_json:
        if data_ann['question_normal'] == question:
            for entity_dict in data_ann['node_mention_nju']:
                if entity_dict['tag'] == 'entity':
                    entity_list.append(entity_dict['uri'])
            break
    return entity_list


def get_topic_literals_by_question(question=None):
    entity_list = []
    for data_ann in annotation_node_questions_json:
        if data_ann['question_normal'] == question:
            for entity_dict in data_ann['node_mention_nju']:
                if entity_dict['tag'] == 'literal':
                    entity_list.append(entity_dict['uri'])
    return entity_list


def get_topic_entities_by_question_and_mention(question=None, mention=None):
    result = dict()
    for data_ann in annotation_node_questions_json:
        if data_ann['question_normal'] == question:
            for entity_dict in data_ann['node_mention_nju']:
                if entity_dict['mention'] == mention:
                    result[entity_dict['uri']] = 1.0
                    break
    return result


def get_grounded_graph_by_question(question=None):
    gold_grounded_graph = None
    for data_ann in test_graph_questions_struct:
        if data_ann.question == question:
            gold_grounded_graph = GrounedGraph(nodes=data_ann.nodes, edges=data_ann.edges)
            break
    for data_ann in train_graph_questions_struct:
        if data_ann.question == question:
            gold_grounded_graph = GrounedGraph(nodes=data_ann.nodes, edges=data_ann.edges)
    return gold_grounded_graph


def get_gold_graph_query_by_question(question=None):
    gold_graph_query = None
    for data_ann in test_graph_questions_struct:
        if data_ann.question == question:
            gold_graph_query = data_ann.graph_query
            break
    for data_ann in train_graph_questions_struct:
        if data_ann.question == question:
            gold_graph_query = data_ann.graph_query
    return gold_graph_query


def get_sparql_by_question(question=None):
    sparql = None
    for data_ann in test_graph_questions_struct:
        if data_ann.question == question:
            sparql = data_ann.sparql_query
            break
    for data_ann in train_graph_questions_struct:
        if data_ann.question == question:
            sparql = data_ann.sparql_query
    return sparql


def get_q_aggregation_type_by_question(question=None):
    _function = []
    for data_ann in test_graph_questions_struct:
        if data_ann.question == question:
            _function = data_ann.function
            break
    for data_ann in train_graph_questions_struct:
        if data_ann.question == question:
            _function = data_ann.function
    return _function


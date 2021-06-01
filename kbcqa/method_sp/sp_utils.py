
import unicodedata
import numpy as np


def convert_question_to_question(question):
    return unicodedata.normalize('NFKD', question).encode('ascii', 'ignore').decode(encoding='UTF-8')


def is_question_node(ungrounded_node):
    '''check if node is class node'''
    class_list = ["class"]
    if ungrounded_node.node_type in class_list and ungrounded_node.question_node == 1:
        return True
    else:
        return False


def search_question_node_nid(nodes):
    '''question node index'''
    question_node_index = None
    for node in nodes:
        if is_question_node(node):
            question_node_index = node.nid
    return question_node_index


def get_gold_entity_or_class(gold_grounded_graph, node_type='entity'):
    gold_set_result = set()
    if gold_grounded_graph is None:
        return gold_set_result
    for gold_node in gold_grounded_graph.nodes:
        if gold_node.node_type == node_type:
            gold_set_result.add(gold_node.id)
    return gold_set_result


def get_gold_entity_or_class_by_json(gold_grounded_graph, node_type='entity'):
    gold_set_result = set()
    if gold_grounded_graph is None:
        return gold_set_result
    for gold_node in gold_grounded_graph['nodes']:
        if gold_node['node_type'] == node_type:
            gold_set_result.add(gold_node['id'])
    return gold_set_result


def get_nid_by_id(nodes, id):
    result_nid = None
    for node in nodes:
        if id == node.id:
            result_nid = node.nid
    return result_nid


def get_edge_by_nodes(edges, node_a, node_b):
    result = None
    for edge in edges:
        if (edge.start == node_a.nid and edge.end == node_b.nid) or (edge.start == node_b.nid and edge.end == node_a.nid):
            result = edge
    return result


def is_literal_node(nodes, nid):
    result = False
    for node in nodes:
        if (nid == node.nid and node.node_type == 'literal') or nid in ['?num', '?sk0']:
            result = True
    return result


def get_unground_node_by_id(nodes, id):
    result_node = None
    for node in nodes:
        if id == node.nid:
            result_node = node
            break
    return result_node


def Normalize(data):
    if len(data)==0:
        return []
    m = np.mean(data)
    mx = max(data)
    mn = min(data)
    return [(float(i) - m) / (mx - mn) for i in data]


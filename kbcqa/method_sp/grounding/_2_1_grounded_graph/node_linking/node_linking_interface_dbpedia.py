import collections
from collections import OrderedDict
from method_sp.grounding._2_1_grounded_graph.node_linking import node_linking_args, node_linking_utils


def node_linking(qid=None, node=None, top_k=None, elp=None):
    '''node grounding
    {'entity_id', 'entity_pro'}
    return top_k results'''
    results_dict = OrderedDict()
    if node.node_type == 'entity':
        entities_pros = el_interface(mention=node_linking_utils.get_old_mention(node.friendly_name), topk=10)
        for entity, pro in entities_pros.items():
            results_dict[entity] = pro
    elif node.node_type == 'class':
        results_dict['hhh'] = 1.0 # results_dict = cl_interface(node=node) 2021.01.30
    elif node.node_type == 'literal' or node.node_type == 'DATE':
        if node.normalization_value is not None:
            results_dict[node.normalization_value] = 1.0
        else:
            results_dict[node.friendly_name] = 1.0
        node.node_type = 'literal'
    else:
        pass
    return results_dict


def el_interface(mention, topk=10):
    """mention='President Lincoln'"""
    """labels"""
    labels_iris_dict = el_labels(mention=mention)
    """wikipage redirect"""
    wikipage_iris_dict = el_wikiPageRedirects(mention=mention)
    """wikiLinkText"""
    wikilinkText_iris_dict =el_wikiLinkText(mention=mention)
    """merge and rerank"""
    merge_dict = dict()
    merge_dict = node_linking_utils.add_dict_number(merge_dict, labels_iris_dict)
    merge_dict = node_linking_utils.add_dict_number(merge_dict, wikipage_iris_dict)
    merge_dict = node_linking_utils.add_dict_number(merge_dict, wikilinkText_iris_dict)
    entities_tuple_list = sorted(merge_dict.items(), key=lambda d:d[1], reverse=True)
    result_entities_dict = collections.OrderedDict()
    for i, (entity_id, surface_score) in enumerate(entities_tuple_list):
        i += 1
        result_entities_dict[entity_id] = surface_score
        if i >= topk:
            break
    return rerank_by_degree(result_entities_dict)  # result_entities_dict


def el_labels(mention):
    return _link(mention=mention, label_to_iris_dict_dict=node_linking_args.label_to_iris_dict_dict, weight=0.5)


def el_wikiPageRedirects(mention):
    return _link(mention=mention, label_to_iris_dict_dict=node_linking_args.wikipage_to_iris_dict_dict, weight=0.3)


def el_wikiLinkText(mention):
    return _link(mention=mention, label_to_iris_dict_dict=node_linking_args.wikilinkText_to_iris_dict_dict, weight=0.2)


def _link(mention='Google Videos', label_to_iris_dict_dict=None, weight=1.0):
    iris_dict = dict()
    mention = mention.lower()
    if mention in label_to_iris_dict_dict:
       iris_dict = label_to_iris_dict_dict[mention]
    for iri, score in iris_dict.items():
        iris_dict[iri] = weight
    return iris_dict


def rerank_by_degree(entities_dict):
    from datasets_interface.virtuoso_interface.dbpedia_kb_interface import get_out_edge_degree
    temp_entities_dict = collections.OrderedDict()
    for entity_id, surface_score in entities_dict.items():
        degree = get_out_edge_degree(entity_id)
        if len(degree) > 0:
            degree = int(degree.pop())
        else:
            degree = 0
        temp_entities_dict[entity_id] = degree * surface_score
    entities_tuple_list = sorted(temp_entities_dict.items(), key=lambda d:d[1], reverse=True)
    result_entities_dict = collections.OrderedDict()
    for i, (entity_id, degree_score) in enumerate(entities_tuple_list):
        i += 1
        result_entities_dict[entity_id] = degree_score
    return result_entities_dict


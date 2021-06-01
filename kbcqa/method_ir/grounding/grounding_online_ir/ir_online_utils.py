from method_sp.parsing import parsing_utils
import copy
from method_ir.grounding.grounding_online_ir.candidate_generation_args import kb_prefix, kb_type_predicate_list


def get_triples_by_grounded_graph_edges(nodes, edges):
    triples = []
    for edge in edges:
        start_node = parsing_utils.search_one_node_in_nodes_by_nid(nodes, edge.start)
        end_node = parsing_utils.search_one_node_in_nodes_by_nid(nodes, edge.end)
        if start_node.node_type == 'entity':
            start_node_uri = kb_prefix + start_node.id
        else:
            if start_node.question_node == 1:
                start_node_uri = '?answer_'+str(start_node.nid)
            else:
                start_node_uri = '?c_'+str(start_node.nid)
        if end_node.node_type == 'entity':
            end_node_uri = kb_prefix + end_node.id
        else:
            if end_node.question_node == 1:
                end_node_uri = '?answer_' + str(end_node.nid)
            else:
                end_node_uri = '?c_'+str(end_node.nid)
        triple = dict()
        triple['subject'] = start_node_uri
        triple['predicate'] = edge.relation
        triple['object'] = end_node_uri
        triples.append(triple)
    return rerank_triples(triples)


def get_triples_by_sparql_json(sparql_json, _type='bgp'):
    triples = []
    if sparql_json == 'parse_error':
        return triples
    for temp_triples in sparql_json['where']:
        if _type == temp_triples['type']:
            for triple in temp_triples['triples']:
                if triple['predicate'] not in kb_type_predicate_list:
                    triples.append(copy.deepcopy(triple))
        elif temp_triples['type'] == 'query':
            for triple in get_triples_by_sparql_json(sparql_json=temp_triples, _type=_type):
                triples.append(copy.deepcopy(triple))
    return rerank_triples(triples)


def convert_triples_to_path(triples):
    #3 triples to path
    path = []
    for triple in triples:
        if is_dbpedia_uri(triple['subject']):
            path.append('+' + triple['predicate'])
        elif is_dbpedia_uri(triple['object']):
            path.append('-' + triple['predicate'])
        # elif triple['subject'] == '?c' and triple['object'] == '?a':
        elif '?answer' in triple['object'] or '?uri' in triple['object']:
            path.append('+' + triple['predicate'])
        else:
            path.append('-' + triple['predicate'])
    return handle_path(path)


def get_question_type_by_sparql_json(sparql_json):
    '''get question type from sparql json data SELECT, ASK'''
    q_type = 'SELECT'
    if 'queryType' in sparql_json:
        q_type = sparql_json['queryType']
    return q_type


def get_aggregation_function_by_sparql_json(sparq_json):
    aggregation_function = 'none'
    if 'variables' in sparq_json:
        for variable in sparq_json['variables']:
            if 'expression' in variable:
                if 'aggregation' in variable['expression']:
                    aggregation_function = variable['expression']['aggregation']
                    break
    return aggregation_function


def get_type_constraints_by_sparql_json(sparql_json):
    '''variable: type_uri'''
    type_constraints = {}
    if sparql_json == 'parse_error':
        return type_constraints
    for graph_pattern in sparql_json['where']:
        if graph_pattern['type'] == 'bgp':
            for triple in graph_pattern['triples']:
                if triple['predicate'] in kb_type_predicate_list:
                    try:
                        if triple['subject'] in sparql_json['variables']:
                            type_constraints['?uri'] = triple['object']  # The constraint is on the uri
                        else:
                            type_constraints['?x'] = triple['object']
                    except KeyError:
                        type_constraints['?x'] = triple['object']
        else:
            print(graph_pattern['type'])
    return type_constraints


def is_exist_gold_path(hop_list, gold_path):
    ''' True 代表不存在positive  False 代表存在positive'''
    no_positive_path = True
    for hop_dict in hop_list:
        if hop_dict['path'] == gold_path:
            no_positive_path = False  # False 代表存在positive
            break
    return no_positive_path


def topic_entities_with_t(entities_list):
    topic_entities_list = []
    for entity in entities_list:
        if entity is not None:
            topic_entities_list.append([entity.replace(kb_prefix, ''), 'entity'])
    return topic_entities_list


def rerank_triples(triples):
    '''对triples集合重新排序
    优先顺序e > l > c > a     当有相同类型时候, 考虑字母顺序
    '''
    def _get_triple_by_entity_node(triples, node):
        triple = None
        for temp in triples:
            if node == temp['subject'] or node == temp['object']:
                triple = temp
        return triple

    def _get_triple_by_subject_class(triples, subject_class):
        triple = None
        for temp in triples:
            if subject_class == temp['subject'] and not is_dbpedia_uri(temp['object']):
                triple = temp
        return triple

    new_triples = []
    node_list = []
    for triple in triples:
        if is_dbpedia_uri(triple['subject']):
            node_list.append(triple['subject'])
        elif is_dbpedia_uri(triple['object']):
            node_list.append(triple['object'])
        else:
            node_list.append(triple['subject'])

    # sorted
    node_list = sorted(node_list)
    # add entity
    for node in node_list:
        if is_dbpedia_uri(node):
            new_triples.append(_get_triple_by_entity_node(triples=triples, node=node))
    # add class
    for node in node_list:
        if not is_dbpedia_uri(node):
            new_triples.append(_get_triple_by_subject_class(triples=triples, subject_class=node))
    return new_triples


def is_dbpedia_uri(_string):
    if _string.startswith('http://dbpedia.org/'):
        return True
    elif _string.startswith('http://rdf.freebase.com/ns'):
        return True
    return False


def get_literal_num(entity_or_literal_with_type_list):
    '''count literal node'''
    literal_num = 0
    for entity_or_literal_with_type in entity_or_literal_with_type_list:
        if entity_or_literal_with_type[1] == 'literal':
            literal_num += 1
    return literal_num


def get_literal_and_entity_from_list(entity_or_literal_with_type_list):
    e1 = None
    l2 = None
    for entity_or_literal_with_type in entity_or_literal_with_type_list:
        if entity_or_literal_with_type[1] == 'literal':
            l2 = entity_or_literal_with_type[0]
        elif entity_or_literal_with_type[1] == 'entity':
            e1 = entity_or_literal_with_type[0]
    return e1, l2


def merge(dict1, dict2):
    new_dict = copy.deepcopy(dict1)
    new_dict.update(dict2)
    return new_dict


def get_entity1_and_entity2_from_list(entity_or_literal_with_type_list):
    # sorted
    node_list = []
    for entity_or_literal_with_type in entity_or_literal_with_type_list:
        node_list.append(entity_or_literal_with_type[0])
    node_list = sorted(node_list)
    return node_list[0], node_list[1]


def handle_path(path):
    '''
    :param path: ['-http://dbpedia.org/property/mother', '+http://dbpedia.org/property/spouse']
    :return: ['-','http://dbpedia.org/property/mother', '+', 'http://dbpedia.org/property/spouse']
    '''
    new_path = []
    for p in path:
        new_path.append(p[0])
        new_path.append(p[1:])
    return new_path


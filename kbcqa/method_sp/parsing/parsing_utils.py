from collections import defaultdict
from nltk.parse import DependencyGraph
import copy
from common_structs.skeleton import Token
from method_sp.parsing import parsing_args
from method_sp import sp_utils


def create_tokens(words):
    '''create node token'''
    # return [Node(index, token) for index, token in enumerate(tokens)]
    tokens = []
    for index, word in enumerate(words):
        if index == len(words)-1 :
            tokens.append(Token(index, word, isEnd=True))
        else:
            tokens.append(Token(index, word, isEnd=False))
    return tokens


def is_exist_edge_in_edges(ungrounded_edges, ungrounded_edge):
    result = False
    for edge in ungrounded_edges:
        if (edge.start == ungrounded_edge.start and edge.end == ungrounded_edge.end) \
                or (edge.start == ungrounded_edge.end and edge.end == ungrounded_edge.start):
            result = True
            break
    return result


def is_contained_one_node_from_nodes(token_index, ungrounded_nodes):
    result = False
    for ungrounded_node in ungrounded_nodes:
        if ungrounded_node.start_position <= token_index <= ungrounded_node.end_position:
            result = True
            break
    return result


def look_for_one_node_from_nodes(token_index, ungrounded_nodes):
    result = None
    for ungrounded_node in ungrounded_nodes:
        if ungrounded_node.start_position <= token_index <= ungrounded_node.end_position:
            result = ungrounded_node
            break
    return result


def is_contained_one_node(head_node_index, end_node_index, ungrounded_nodes):
    result = False
    for ungrounded_node in ungrounded_nodes:
        if ungrounded_node.start_position <= head_node_index <= ungrounded_node.end_position \
            and ungrounded_node.start_position <= end_node_index <= ungrounded_node.end_position:
            result = True
            break
    return result


def get_friendly_name_by_dependency(dependency_graph, path):
    '''friendly_name = ' '.join([token.value for token in tokens if sequence_end >= token.index >= sequence_start])'''
    copy_path = copy.deepcopy(path)
    if len(copy_path) > 0: del copy_path[-1]
    friendly_name = ''
    for token_index in sorted(copy_path):
        friendly_name += dependency_graph.nodes[token_index]['word'] + ' '
        # friendly_name += ' '.join([token.value for token in tokens if token_index-1 == token.index])+' '
    # friendly_name += str(copy_path)
    return friendly_name.strip()


def look_for_key_by_value(dict_, value_):
    result = None
    for key, value in dict_.items():
        if value_ == value:
            result = key
            result += 1
            break
    return result


def update_dependencygraph_indexs(old_dependency_graph):
    # surface_tokens_to_dep_node_dict 两个索引的映射表
    surface_tokens_to_dep_node_dict = dict()
    for _, dep_node in old_dependency_graph.nodes.items():
        surface_tokens_to_dep_node_dict[dep_node['feats']] = dep_node['address']

    new_hybrid_dependency_graph = DependencyGraph()
    for _, yuanshi_node in old_dependency_graph.nodes.items():
        new_node_info = copy.deepcopy(yuanshi_node)
        new_node_info['address'] = look_for_key_by_value(dict_=surface_tokens_to_dep_node_dict, value_=new_node_info['address'])
        new_node_info['head'] = look_for_key_by_value(dict_=surface_tokens_to_dep_node_dict, value_=new_node_info['head'])

        new_deps = defaultdict(list)
        for old_dep_rel, old_dep_index_list in yuanshi_node['deps'].items():
            new_dep_index_list = []
            for old_dep_index in old_dep_index_list:
                new_dep_index_list.append(look_for_key_by_value(dict_=surface_tokens_to_dep_node_dict, value_=old_dep_index))
            new_deps[old_dep_rel] = new_dep_index_list
        new_node_info['deps'] = new_deps

        # 更新新地址
        if new_hybrid_dependency_graph.contains_address(new_node_info['address']):
            new_hybrid_dependency_graph.nodes[new_node_info['address']].update(new_node_info)
        else:
            new_hybrid_dependency_graph.add_node(new_node_info)

    if new_hybrid_dependency_graph.nodes[0]['deps']['ROOT']:
        root_address = new_hybrid_dependency_graph.nodes[0]['deps']['ROOT'][0]
        new_hybrid_dependency_graph.root = new_hybrid_dependency_graph.nodes[root_address]
        new_hybrid_dependency_graph.top_relation_label = 'ROOT'
    return new_hybrid_dependency_graph


def class_question_node_class_node_in_one_edge(ungrounded_graph_nodes, edge):
    '''class and class组合成edge'''
    class_question_node = None
    class_node = None
    start_node = search_one_node_in_nodes_by_nid(ungrounded_graph_nodes, edge.start)
    end_node = search_one_node_in_nodes_by_nid(ungrounded_graph_nodes, edge.end)
    if is_class_node(start_node) and sp_utils.is_question_node(start_node) and is_class_node(end_node):
        class_question_node = start_node
        class_node = end_node
    elif is_class_node(end_node) and sp_utils.is_question_node(end_node) and is_class_node(start_node):
        class_question_node = end_node
        class_node = start_node
    return class_question_node, class_node


def is_exist_in_nodes(ungrounded_graph_nodes, ungrounded_node):
    ''''''
    result = False
    for node in ungrounded_graph_nodes:
        if node.nid == ungrounded_node.nid:
            result = True
            break
    return result


def search_one_node_in_nodes(ungrounded_graph_nodes, ungrounded_node):
    result = None
    for node in ungrounded_graph_nodes:
        if node.nid == ungrounded_node.nid:
            result = node
            break
    return result


def search_one_node_in_nodes_by_nid(ungrounded_graph_nodes, nid):
    result = None
    for node in ungrounded_graph_nodes:
        if node.nid == nid:
            result = node
            break
    return result


def is_class_node(ungrounded_node):
    '''check if node is class node'''
    class_list = ["class"]
    if ungrounded_node.node_type in class_list:
        return True
    else:
        return False


def is_entity_node(ungrounded_node):
    '''check if node is entity node'''
    class_list = ["entity"]
    if ungrounded_node.node_type in class_list:
        return True
    else:
        return False


def search_adjacent_edges(node, ungrounded_graph):
    ''''''
    adjacent_edges = []
    for edge in ungrounded_graph.edges:
        if edge.start == node.nid or edge.end == node.nid:
            adjacent_edges.append(edge)
    return adjacent_edges


def del_edge_in_ungrounded_edge(ungrounded_graph, directed_cycle):
    '''
    :param ungrounded_graph:
    :param directed_cycle:
    :return: ungrounded_graph
    '''
    ungrounded_graph = copy.deepcopy(ungrounded_graph)

    edge_nodes_pair = []
    edge_nodes_pair.append(('literal', 'entity'))
    edge_nodes_pair.append(('entity', 'literal'))
    edge_nodes_pair.append(('literal', 'literal'))
    edge_nodes_pair.append(('entity', 'entity'))

    # node_type
    ungrounded_graph_nodes = ungrounded_graph.nodes
    ungrounded_graph_edges = ungrounded_graph.edges
    remove_edges = []

    for cycle in directed_cycle.all_cycles:
        cycle_index_list = []
        for temp_index in cycle:
            cycle_index_list.append(temp_index)

        before_node_index = None
        current_node_index = None
        for i, index in enumerate(cycle_index_list):
            if i == 0:
                before_node_index = index
                continue
            else:
                current_node_index = index
                temp_edge = search_special_edge_in_edges(ungrounded_graph_edges, before_node_index, current_node_index)
                if temp_edge is not None:
                    before_node = search_one_node_in_nodes_by_nid(ungrounded_graph_nodes, before_node_index)
                    current_node = search_one_node_in_nodes_by_nid(ungrounded_graph_nodes, current_node_index)
                    if is_nosiy_edge(edge_nodes_pair=edge_nodes_pair, node_a=before_node, node_b=current_node):
                        remove_edges.append(temp_edge)
                before_node_index = current_node_index

        if len(remove_edges) == 0:
            #if class_1 - class_2{question_node=1} - entity|literal
            #断开 class_2{question_node} - entity|literal
            before_node_index = None
            for i, index in enumerate(cycle_index_list):
                if i == 0:
                    before_node_index = index
                    continue
                else:
                    current_node_index = index
                    current_node = search_one_node_in_nodes_by_nid(ungrounded_graph_nodes, current_node_index)
                    if current_node.node_type == 'entity' or current_node.node_type == 'literal':
                        before_edge = search_special_edge_in_edges(ungrounded_graph_edges, before_node_index, current_node_index)
                        before_node = search_one_node_in_nodes_by_nid(ungrounded_graph_nodes, before_node_index)
                        if before_edge is not None and before_node.node_type == 'class':

                            if len(cycle_index_list) > i+1: # 未到达了最后一条边
                                next_edge = search_special_edge_in_edges(ungrounded_graph_edges, current_node_index, cycle_index_list[i+1])
                                next_node = search_one_node_in_nodes_by_nid(ungrounded_graph_nodes, cycle_index_list[i+1])
                                if next_edge is not None and next_node.node_type == 'class':
                                    if before_node.question_node == 1 and next_node.question_node == 0:
                                        remove_edges.append(before_edge)
                                    elif before_node.question_node == 0 and next_node.question_node == 1:
                                        remove_edges.append(next_edge)

                            else: #到达了最后一条边
                                next_edge = search_special_edge_in_edges(ungrounded_graph_edges, cycle_index_list[0], cycle_index_list[1])
                                next_node = search_one_node_in_nodes_by_nid(ungrounded_graph_nodes, cycle_index_list[1])
                                if next_edge is not None and next_node.node_type == 'class':
                                    if before_node.question_node == 1 and next_node.question_node == 0:
                                        remove_edges.append(before_edge)
                                    elif before_node.question_node == 0 and next_node.question_node == 1:
                                        remove_edges.append(next_edge)

                    before_node_index = current_node_index
        break  #default one cycle

    new_ungrounded_edges = []
    for edge in ungrounded_graph.edges:
        if edge not in remove_edges:
            new_ungrounded_edges.append(copy.deepcopy(edge))
    ungrounded_graph.edges = new_ungrounded_edges
    ungrounded_graph.ungrounded_query_id = ungrounded_graph.ungrounded_query_id + 1
    return ungrounded_graph


def search_special_edge_in_edges(ungrounded_edges, nid_a, nid_b):
    result = None
    for edge in ungrounded_edges:
        if (edge.start == nid_a and edge.end == nid_b) or (edge.start == nid_b and edge.end == nid_a):
            result = edge
            break
    return result


def is_nosiy_edge(edge_nodes_pair, node_a, node_b):
    result = False
    for edge_node_pair_start, edge_node_pair_end in edge_nodes_pair:
        if node_a.node_type == edge_node_pair_start and node_b.node_type == edge_node_pair_end:
            result = True
    return result


def abstract_question_word_generation(tokens, ungrounded_nodes):
    '''
    replace entity mention and class mention to entity and class
    '''
    i = 0
    abstract_question_word = []
    while i < len(tokens):
        is_contained = False
        # for sequence_start_end, ner_tag in sequence_ner_tag_dict.items():
        #     if ner_tag not in ['entity', 'literal', 'class']: continue
        #     sequence_start = int(sequence_start_end.split('\t')[0])
        #     sequence_end = int(sequence_start_end.split('\t')[1])
        #     if sequence_start <= i <= sequence_end:
        #         # self.abstract_question_word.append('NP')
        #         # self.abstract_question_pos.append('NP')
        #         abstract_question_word.append(ner_tag)
        #         # self.abstract_question_pos.append(node_type)
        #         is_contained = True
        #         i += (sequence_end - sequence_start + 1)
        #         break
        for ungrounded_node in ungrounded_nodes:
            sequence_start = ungrounded_node.start_position
            sequence_end = ungrounded_node.end_position
            ner_tag = ungrounded_node.node_type
            if ner_tag not in ['entity', 'literal', 'class']:
                continue
            if sequence_start <= i <= sequence_end:
                abstract_question_word.append(ner_tag)
                is_contained = True
                i += (sequence_end - sequence_start + 1)
                break
        if not is_contained:
            abstract_question_word.append(tokens[i].value)
            i += 1
    return abstract_question_word


def search_for_node_by_index(node_index, dependency_graph):
    result_node = None
    for index, node in dependency_graph.nodes.items():
        if index == node_index:
            result_node = node
    return result_node


def adj_edge_nodes_update(dep_node_index, dependency_graph):
    '''zhao children'''
    dep_result_nodes_index = []
    if dep_node_index is None:
        return dep_result_nodes_index
    dep_node = search_for_node_by_index(dep_node_index, dependency_graph)
    #chu du
    for _, child_index_list in dep_node['deps'].items():
        for child_index in child_index_list:
            dep_result_nodes_index.append(child_index)

    #ru du
    for node_index_, node in dependency_graph.nodes.items():
        for _, child_index_list in node['deps'].items():
            for child_index in child_index_list:
                if child_index == dep_node_index: #某个顶点的孩子是dep_node, 那么他就是其父亲
                    dep_result_nodes_index.append(node_index_)
    return dep_result_nodes_index


def get_nertag_sequence(ungrounded_nodes):
    indexs_to_nertag_dict = dict()
    for ungrounded_node in ungrounded_nodes:
        sequence_start = ungrounded_node.start_position
        sequence_end = ungrounded_node.end_position
        ner_tag = ungrounded_node.node_type
        indexs_to_nertag_dict[str(sequence_start) + '\t' + str(sequence_end)] = ner_tag
    return indexs_to_nertag_dict


def importantwords_by_unimportant_abstractq(abstractquestion):
    if isinstance(abstractquestion, list):
        abstractquestion = ' '.join(abstractquestion)
    abstractquestion_remove_version = abstractquestion.lower()
    for unimportantphrase in parsing_args.unimportantphrases:
        abstractquestion_remove_version = abstractquestion_remove_version.replace(unimportantphrase, "")
    importantwords = []
    for word in abstractquestion_remove_version.split(" "):
        if len(word) > 0 \
                and word not in parsing_args.unimportantwords \
                and word not in parsing_args.stopwords_dict:
            importantwords.append(word)
    # importantwords = set(abstractquestion_remove_version.split(" ")) - unimportantwords - stopwords
    # for word in importantwords:
    #     if "entity" in word:
    #         print("error")
    return importantwords


def extract_importantwords_from_question(question, ungrounded_graph):
    for node in ungrounded_graph.nodes:
        if node.node_type == 'entity':
            question = question.replace(node.friendly_name, '')
    words = question.split(' ')
    if words[-1] == '?' or words[-1] == '.':
        del words[-1]

    abstractquestion_list = []  #变成了list
    for word in words:
        if len(word) > 0:
            abstractquestion_list.append(word)
    return importantwords_by_unimportant_abstractq(abstractquestion_list)


def get_importantwords_byabstractquestion(question_):
    words = question_.split()
    if len(words)==0:
        return []
    if words[-1] == '?' or words[-1] == '.':
        del words[-1]
    importantwords_list = list()
    for word in words:
        if len(word) > 0 and '<e>' not in word:
            importantwords_list.append(word)
    return importantwords_by_unimportant_abstractq(importantwords_list)

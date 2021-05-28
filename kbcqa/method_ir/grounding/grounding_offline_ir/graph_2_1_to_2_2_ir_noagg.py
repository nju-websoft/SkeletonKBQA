from common.hand_files import read_json
from method_sp.grounding._2_2_grounded_graph.grounding_offline_sp import path_to_graph_bgp
from method_sp.grounding import grounding_args
from method_sp.grounding import grounding_utils


#4
def _convert_file_to_oracle_graphs(file_path, question_type, entities_or_literals):
    candidate_graphquerys = []
    data_dict = read_json(grounding_args.oracle_file_root + file_path)
    if question_type == 'composition':
        if grounding_args.q_mode in ['cwq', 'graphq']:
            candidate_graphquerys = path_to_graph_bgp.parser_composition_q_freebase_ir(data_dict=data_dict, s1=entities_or_literals[0][0], t1=entities_or_literals[0][1])
        elif grounding_args.q_mode in ['lcquad']:
            candidate_graphquerys = path_to_graph_bgp.parser_composition_q_dbpedia_ir(data_dict=data_dict, s1=entities_or_literals[0][0], t1=entities_or_literals[0][1])
    elif question_type == 'conjunction':
        if grounding_args.q_mode in ['cwq', 'graphq']:
            candidate_graphquerys = path_to_graph_bgp.parser_conjunction_q_freebase(data_dict=data_dict,
                s1=entities_or_literals[0][0], t1=entities_or_literals[0][1],
                s2=entities_or_literals[1][0], t2=entities_or_literals[1][1])
        elif grounding_args.q_mode in ['lcquad']:
            candidate_graphquerys.extend(path_to_graph_bgp.parser_conjunction_q_dbpedia(data_dict=data_dict,
                s1=entities_or_literals[0][0], t1=entities_or_literals[0][1],
                s2=entities_or_literals[1][0], t2=entities_or_literals[1][1]))
            candidate_graphquerys.extend(path_to_graph_bgp.parser_yesno_q_dbpedia(data_dict=data_dict,
                s1=entities_or_literals[0][0], t1=entities_or_literals[0][1],
                s2=entities_or_literals[1][0], t2=entities_or_literals[1][1]))
    return grounding_utils.candidate_query_to_grounded_graph(candidate_graphquerys=candidate_graphquerys)


#3
def _get_2_2_graphs_by_type_and_literals(question_type, entities_or_literals):
    candidate_graphquerys = []
    if len(entities_or_literals) == 2: # one literal, one entity
        literal_value = None
        entity_value = None
        for entity_or_literal in entities_or_literals:
            if entity_or_literal[1] == 'literal':
                literal_value = entity_or_literal[0]
            else:
                entity_value = entity_or_literal[0]
        if not isinstance(literal_value, str):
            literal_value = str(literal_value)

        literal_value = grounding_utils.literal_postprocess(literal_value, q_mode=grounding_args.q_mode)
        if literal_value in grounding_args.literal_to_id_map:
            literal_value_id = str(grounding_args.literal_to_id_map[literal_value])
        else:
            return []

        filename_1 = question_type + '_entity_' + entity_value + '_literal_' + literal_value_id
        filename_2 = question_type + '_literal_' + literal_value_id + '_entity_' + entity_value
        if filename_1 in grounding_args.oracle_all_files_path_names:
            data_dict = read_json(grounding_args.oracle_file_root + filename_1)
            if question_type == 'conjunction':
                if grounding_args.q_mode in ['cwq', 'graphq']:
                    candidate_graphquerys = path_to_graph_bgp.parser_conjunction_q_freebase(data_dict=data_dict, s1=entity_value, t1='entity', s2=literal_value_id, t2='literal')
                elif grounding_args.q_mode in ['lcquad']:
                    candidate_graphquerys = path_to_graph_bgp.parser_conjunction_q_dbpedia(data_dict=data_dict, s1=entity_value, t1='entity', s2=literal_value_id, t2='literal')
        elif filename_2 in grounding_args.oracle_all_files_path_names:
            data_dict = read_json(grounding_args.oracle_file_root + filename_2)
            if grounding_args.q_mode in ['cwq', 'graphq']:
                candidate_graphquerys = path_to_graph_bgp.parser_conjunction_q_freebase(data_dict=data_dict, s1=literal_value_id, t1='literal', s2=entity_value, t2='entity')
            elif grounding_args.q_mode in ['lcquad']:
                candidate_graphquerys = path_to_graph_bgp.parser_conjunction_q_dbpedia(data_dict=data_dict, s1=literal_value_id, t1='literal', s2=entity_value, t2='entity')

    elif len(entities_or_literals) == 1:
        literal_value = None
        for entity_or_literal in entities_or_literals:
            if entity_or_literal[1] == 'literal':
                literal_value = entity_or_literal[0]
        if not isinstance(literal_value, str):
            literal_value = str(literal_value)

        literal_value = grounding_utils.literal_postprocess(literal_value, q_mode=grounding_args.q_mode)
        if literal_value in grounding_args.literal_to_id_map:
            literal_value_id = str(grounding_args.literal_to_id_map[literal_value])
        else:
            return []

        filename_1 = question_type + '_literal_' + literal_value_id
        if filename_1 in grounding_args.oracle_all_files_path_names:
            data_dict = read_json(grounding_args.oracle_file_root + filename_1)
            if question_type == 'composition':
                if grounding_args.q_mode in ['cwq', 'graphq']:
                    candidate_graphquerys = path_to_graph_bgp.parser_composition_q_freebase_ir(data_dict=data_dict, s1=literal_value, t1='literal')
                elif grounding_args.q_mode in ['lcquad']:
                    candidate_graphquerys = path_to_graph_bgp.parser_composition_q_dbpedia_ir(data_dict=data_dict, s1=literal_value, t1='literal')

    return grounding_utils.candidate_query_to_grounded_graph(candidate_graphquerys=candidate_graphquerys)


#3
def _get_oracle_graphs_by_type_and_entities(question_type, entities_or_literals):
    '''
    # 'conjunction_entity_m.0345h_entity_m.0177z': continue
    # 'composition_entity_m.021hz5': continue
    :param question_type:
    :param entities_or_literals:
    :return: oracle grounded graphs
    '''
    filename_1, filename_2 = grounding_utils.get_local_filename(entities_or_literals)
    filename_1 = question_type + '_' + filename_1
    if filename_2 is not None:
        filename_2 = question_type + '_' + filename_2
    grounded_graph_list = []
    if filename_1 in grounding_args.oracle_all_files_path_names:
        grounded_graph_list = _convert_file_to_oracle_graphs(file_path=filename_1, question_type=question_type, entities_or_literals=entities_or_literals)
    elif filename_2 is not None and filename_2 in grounding_args.oracle_all_files_path_names:
        new_entities_or_literals = entities_or_literals
        if len(entities_or_literals) >= 2:
            new_entities_or_literals.append(entities_or_literals[1])
            new_entities_or_literals.append(entities_or_literals[0])
        grounded_graph_list = _convert_file_to_oracle_graphs(file_path=filename_2, question_type=question_type, entities_or_literals=new_entities_or_literals)
    return grounded_graph_list


#1
def get_oracle_graphs_by_2_1_graph(_2_1_grounded_graph):
    '''
    zong's interface, 基于ie的转换方式
    :param _2_1_ungrounded_graph:
    :return: oracle graphs
    '''
    entities_list = []
    literals_list = []
    for node in _2_1_grounded_graph.nodes:
        if node.node_type == 'entity':
            entities_list.append([node.id.replace('http://dbpedia.org/resource/', ''), node.node_type])
        elif node.node_type == 'literal':
            literals_list.append([node.id,node.node_type])

    oracle_graphs = []
    if len(entities_list) == 1 and len(literals_list) == 1:
        print('#literal!!!', entities_list, literals_list)
        entities_list.append(literals_list[0])
        oracle_graphs = _get_2_2_graphs_by_type_and_literals(question_type='conjunction', entities_or_literals=entities_list)
    elif len(entities_list) == 2:
        oracle_graphs = _get_oracle_graphs_by_type_and_entities(question_type='conjunction', entities_or_literals=entities_list)
    elif len(entities_list) == 1:
        oracle_graphs = _get_oracle_graphs_by_type_and_entities(question_type='composition', entities_or_literals=entities_list)
    elif len(literals_list) == 1:
        print('#literal!!!', entities_list, literals_list)
        oracle_graphs = _get_2_2_graphs_by_type_and_literals(question_type='composition', entities_or_literals=literals_list)
    return oracle_graphs


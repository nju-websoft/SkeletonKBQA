from method_sp.grounding._2_2_grounded_graph.grounding_offline_sp import path_to_graph_bgp
from common.hand_files import read_json
from method_sp.grounding import grounding_utils
from method_sp.grounding import grounding_args


#3
def _convert_file_to_oracle_graphs(file_path, question_type, entities_or_literals, is_constraint_mediator=False):
    candidate_graphquerys = []
    data_dict = read_json(grounding_args.oracle_file_root+file_path)
    if question_type == 'composition':
        if grounding_args.q_mode in ['cwq', 'graphq']:
            candidate_graphquerys = path_to_graph_bgp.parser_composition_q_freebase_sp(
                data_dict=data_dict,
                s1=entities_or_literals[0][0], t1=entities_or_literals[0][1], is_constraint_mediator=is_constraint_mediator)
        elif grounding_args.q_mode in ['lcquad']:
            candidate_graphquerys = path_to_graph_bgp.parser_composition_q_dbpedia_sp(
                data_dict=data_dict,
                s1=entities_or_literals[0][0], t1=entities_or_literals[0][1], is_constraint_mediator=is_constraint_mediator)
    elif question_type == 'conjunction':
        if grounding_args.q_mode in ['cwq', 'graphq']:
            candidate_graphquerys = path_to_graph_bgp.parser_conjunction_q_freebase(
                data_dict=data_dict,
                s1=entities_or_literals[0][0], t1=entities_or_literals[0][1],
                s2=entities_or_literals[1][0], t2=entities_or_literals[1][1])
        elif grounding_args.q_mode in ['lcquad']:
            candidate_graphquerys = path_to_graph_bgp.parser_conjunction_q_dbpedia(
                data_dict=data_dict,
                s1=entities_or_literals[0][0], t1=entities_or_literals[0][1],
                s2=entities_or_literals[1][0], t2=entities_or_literals[1][1])
    elif question_type == 'ask':
        if grounding_args.q_mode in ['lcquad']:
            candidate_graphquerys = path_to_graph_bgp.parser_yesno_q_dbpedia(
                data_dict=data_dict,
                s1=entities_or_literals[0][0], t1=entities_or_literals[0][1],
                s2=entities_or_literals[1][0], t2=entities_or_literals[1][1])

    return grounding_utils.candidate_query_to_grounded_graph(candidate_graphquerys=candidate_graphquerys)


#2
def _get_2_2_graphs_by_structure_and_type_only_entities(question_type=None, entities_or_literals=None, is_constraint_mediator=False):
    if question_type in ['ask', 'conjunction']:
        prefix = 'conjunction'
    else:
        prefix = question_type
    filename_1, filename_2 = grounding_utils.get_local_filename(entities_or_literals)
    filename_1 = prefix + '_' + filename_1
    if filename_2 is not None:
        filename_2 = prefix + '_' + filename_2

    grounded_graph_list = []
    if filename_1 in grounding_args.oracle_all_files_path_names:
        grounded_graph_list = _convert_file_to_oracle_graphs(
            file_path=filename_1, question_type=question_type,
            entities_or_literals=entities_or_literals, is_constraint_mediator=is_constraint_mediator)
    elif filename_2 is not None and filename_2 in grounding_args.oracle_all_files_path_names:
        new_entities_or_literals = entities_or_literals
        if len(entities_or_literals) >= 2:
            new_entities_or_literals.append(entities_or_literals[1])
            new_entities_or_literals.append(entities_or_literals[0])
        grounded_graph_list = _convert_file_to_oracle_graphs(
            file_path=filename_2, question_type=question_type,
            entities_or_literals=new_entities_or_literals, is_constraint_mediator=is_constraint_mediator)

    return grounded_graph_list


#1
def _get_2_2_graphs_by_type_and_literals(question_type, entities_or_literals, is_constraint_mediator=False):
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
                    candidate_graphquerys = path_to_graph_bgp.parser_composition_q_freebase_sp(data_dict=data_dict, s1=literal_value, t1='literal', is_constraint_mediator=is_constraint_mediator)
                elif grounding_args.q_mode in ['lcquad']:
                    candidate_graphquerys = path_to_graph_bgp.parser_composition_q_dbpedia_sp(data_dict=data_dict, s1=literal_value, t1='literal', is_constraint_mediator=is_constraint_mediator)

    return grounding_utils.candidate_query_to_grounded_graph(candidate_graphquerys=candidate_graphquerys)


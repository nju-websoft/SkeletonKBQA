from method_sp.grounding import grounding_utils
from method_sp.grounding._2_2_grounded_graph.grounding_offline_sp import graph_2_1_to_2_2_sp_transfer
from method_ir.grounding.grounding_offline_ir import graph_2_1_to_2_2_ir_agg


def get_oracle_graphs_by_2_1_graph(_2_1_grounded_graph=None):
    '''
    transformation 基于转换的方式
    :param _2_1_grounded_graph:
    :return: _2_2 grounded graphs
    '''
    category = grounding_utils.analysis_structure_category(_2_1_graph=_2_1_grounded_graph)
    """"""
    entities_list = []
    literals_list = []
    for node in _2_1_grounded_graph.nodes:
        if node.node_type == 'entity':
            entities_list.append([node.id.replace('http://dbpedia.org/resource/', ''), node.node_type])
        elif node.node_type == 'literal':
            literals_list.append([node.id, node.node_type])
    """"""
    oracle_graphs = []
    if category == 'composition-0':
        if len(entities_list) == 1:
            oracle_graphs = graph_2_1_to_2_2_sp_transfer._get_2_2_graphs_by_structure_and_type_only_entities(question_type='composition', entities_or_literals=entities_list, is_constraint_mediator=True)
        else:
            print('#only one literal!!!', entities_list, literals_list)
            oracle_graphs = graph_2_1_to_2_2_sp_transfer._get_2_2_graphs_by_type_and_literals(question_type='composition', entities_or_literals=literals_list, is_constraint_mediator=True)

    elif category == 'composition-1':
        if len(entities_list) == 1:
            oracle_graphs = graph_2_1_to_2_2_sp_transfer._get_2_2_graphs_by_structure_and_type_only_entities(question_type='composition', entities_or_literals=entities_list)
        else:
            print('#only one literal!!!', entities_list, literals_list)
            oracle_graphs = graph_2_1_to_2_2_sp_transfer._get_2_2_graphs_by_type_and_literals(question_type='composition', entities_or_literals=literals_list)

    elif category == 'composition-2': #yes-no
        oracle_graphs = graph_2_1_to_2_2_sp_transfer._get_2_2_graphs_by_structure_and_type_only_entities(question_type='ask', entities_or_literals=entities_list)

    elif category == 'conjunction-0':
        if len(entities_list) == 2:
            oracle_graphs = graph_2_1_to_2_2_sp_transfer._get_2_2_graphs_by_structure_and_type_only_entities(question_type='conjunction', entities_or_literals=entities_list)
        elif len(entities_list) == 1 and len(literals_list) == 1:
            print('#literal!!!', entities_list, literals_list)
            entities_list.append(literals_list[0])
            oracle_graphs = graph_2_1_to_2_2_sp_transfer._get_2_2_graphs_by_type_and_literals(question_type='conjunction', entities_or_literals=entities_list)
        else:
            # len(literals_list) == 2: # do not process
            pass

    else:
        print('#other structure!!!!!!', category)
        # oracle_graphs = _2_1_to_2_2_interface.generate_candidates_by_2_1_grounded_graph(_2_1_grounded_graph)
    return oracle_graphs


def get_oracle_graphs_by_2_1_graph_cwq(_2_1_grounded_graph, qtype):
    assert qtype in ['composition', 'conjunction', 'comparative', 'superlative']
    oracle_graphs = []
    if qtype in ['composition', 'conjunction']:
        oracle_graphs = get_oracle_graphs_by_2_1_graph(_2_1_grounded_graph=_2_1_grounded_graph)
    elif qtype in ['comparative', 'superlative']:
        oracle_graphs = graph_2_1_to_2_2_ir_agg._get_oracle_graphs_comparative(_2_1_grounded_graph=_2_1_grounded_graph)
    return oracle_graphs


def get_oracle_graphs_by_2_1_graph_graphq(_2_1_grounded_graph, qtype):
    assert qtype in ['bgp', 'count', 'superlative', 'comparative']
    oracle_graphs = []
    if qtype in ['bgp', 'count']:
        oracle_graphs = get_oracle_graphs_by_2_1_graph(_2_1_grounded_graph=_2_1_grounded_graph)
    elif qtype in ['comparative', 'superlative']:
        oracle_graphs = graph_2_1_to_2_2_ir_agg._get_oracle_graphs_comparative(_2_1_grounded_graph=_2_1_grounded_graph)
    return oracle_graphs



from method_ir.grounding.grounding_offline_ir import graph_2_1_to_2_2_ir_agg, graph_2_1_to_2_2_ir_noagg


def get_oracle_graphs_by_2_1_graph_cwq(_2_1_grounded_graph, qtype):
    assert qtype in ['composition', 'conjunction', 'comparative', 'superlative']
    """按照问句类型"""
    oracle_graphs = []
    if qtype in ['composition', 'conjunction']:
        oracle_graphs = graph_2_1_to_2_2_ir_noagg.get_oracle_graphs_by_2_1_graph(_2_1_grounded_graph=_2_1_grounded_graph)
    elif qtype in ['comparative', 'superlative']:
        oracle_graphs = graph_2_1_to_2_2_ir_agg._get_oracle_graphs_comparative(_2_1_grounded_graph=_2_1_grounded_graph)
    """不按问句类型"""
    # oracle_graphs = graph_2_1_to_2_2_ir_noagg.get_oracle_graphs_by_2_1_graph(_2_1_grounded_graph=_2_1_grounded_graph)
    return oracle_graphs


def get_oracle_graphs_by_2_1_graph_graphq(_2_1_grounded_graph, qtype):
    assert qtype in ['bgp', 'count', 'superlative', 'comparative']
    oracle_graphs = []
    """按照问句类型"""
    if qtype in ['bgp', 'count']:
        oracle_graphs = graph_2_1_to_2_2_ir_noagg.get_oracle_graphs_by_2_1_graph(_2_1_grounded_graph=_2_1_grounded_graph)
    elif qtype in ['comparative', 'superlative']:
        oracle_graphs = graph_2_1_to_2_2_ir_agg._get_oracle_graphs_comparative(_2_1_grounded_graph=_2_1_grounded_graph)
    """不按问句类型"""
    # oracle_graphs = graph_2_1_to_2_2_ir_noagg.get_oracle_graphs_by_2_1_graph(_2_1_grounded_graph=_2_1_grounded_graph)
    return oracle_graphs


def get_oracle_graphs_by_2_1_graph_lcquad(_2_1_grounded_graph, qtype):
    assert qtype in ['bgp', 'count', 'ask']
    oracle_graphs = []
    if qtype in ['bgp', 'count']:
        oracle_graphs = graph_2_1_to_2_2_ir_noagg.get_oracle_graphs_by_2_1_graph(_2_1_grounded_graph=_2_1_grounded_graph)
    elif qtype in ['ask']:
        oracle_graphs = graph_2_1_to_2_2_ir_noagg.get_oracle_graphs_by_2_1_graph(_2_1_grounded_graph=_2_1_grounded_graph)
    return oracle_graphs


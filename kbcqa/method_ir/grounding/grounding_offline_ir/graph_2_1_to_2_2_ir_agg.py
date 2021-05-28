from common.hand_files import read_json
from method_sp.grounding._2_2_grounded_graph.grounding_offline_sp import path_to_graph_comparative
from method_sp.grounding import grounding_args
from method_sp.grounding import grounding_utils


# interface
def _get_oracle_graphs_comparative(_2_1_grounded_graph):
    anchor_entities_list = []
    anchor_literal_list = []
    for node in _2_1_grounded_graph.nodes:
        if node.node_type == 'entity':
            anchor_entities_list.append(node.id)
        elif node.node_type == 'literal':
            anchor_literal_list.append(node.id)
    candidate_graphquerys = []
    for entity in anchor_entities_list:
        print('#anchor:\t',entity)
        filename_1 = 'comparative_entity_' + entity
        if filename_1 in grounding_args.oracle_all_files_path_names:
            data_dict = read_json(grounding_args.oracle_file_root + filename_1)
            candidate_graphquerys.extend(path_to_graph_comparative.parser_comparative_q_freebase_ir(data_dict=data_dict, s1=entity, t1='entity'))
    return grounding_utils.candidate_query_to_grounded_graph(candidate_graphquerys=candidate_graphquerys)


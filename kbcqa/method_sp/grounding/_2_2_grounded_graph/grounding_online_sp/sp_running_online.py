from common_structs.depth_first_search import DepthFirstSearch
from method_sp.grounding._2_2_grounded_graph.grounding_online_sp import sp_candidate_generation_kb, sp_online_utils
from method_sp.grounding._2_2_grounded_graph.grounding_online_sp.construct_grounded_graph import ConstructGroundedGraph
from method_sp.grounding.grounding_args import q_mode
from method_sp.grounding.grounding_utils import literal_postprocess


is_add_mediator = False # True False
mediator_num = 2 #0, 1 or 2
sequence_upper = 10000


def generate_candidates_by_2_1_grounded_graph(_2_1_grounded_graph=None):
    '''
    function: generate candidate grounded graphes
    input: 2.1 structure_with_grounded_graphq_node_grounding_file
    :return: 2.2 candidate grounded graphes
    '''
    grounded_nodes = _2_1_grounded_graph.nodes
    grounded_edges = _2_1_grounded_graph.edges
    #constaint less than 2
    if len(_2_1_grounded_graph.edges) > 2:
        return list()
    if sp_online_utils.has_literal_node(grounded_nodes):
        return list()

    # start node index
    start_vertex = sp_online_utils.search_question_node_nid(grounded_nodes)
    graph = sp_online_utils.convert_triples_to_graph(grounded_edges=grounded_edges)

    # question node is start node, dfs
    depth_first_search = DepthFirstSearch(graph, start_vertex)
    son_father_to_paths_list = []
    nodenid_to_instances_dict = dict()
    for (son_nid, father_nid) in depth_first_search._traversal_sequence_list:
        if father_nid == -1:
            continue

        """1 node and edge"""
        son_node = sp_online_utils.search_one_node_in_nodes_by_nid(ungrounded_graph_nodes=grounded_nodes, nid=son_nid)
        father_node = sp_online_utils.search_one_node_in_nodes_by_nid(ungrounded_graph_nodes=grounded_nodes, nid=father_nid)
        # grounded_edge = online_utils.get_edge_by_nodes(grounded_edges, son_node, father_node)

        """2 generate paths"""
        paths = []
        if is_add_mediator: #consider list, in order to advomcate recall
            paths = list(insert_one_node(son_node=son_node, father_node=father_node, nodenid_to_instances_dict=nodenid_to_instances_dict))

        # grounded_edge.friendly_name
        donot_insert_paths = donot_insert_node(son_node=son_node, father_node=father_node, nodenid_to_instances_dict=nodenid_to_instances_dict)
        for donot_insert_path in donot_insert_paths:
            paths.append(donot_insert_path)
        son_father_to_paths_list.append(((son_nid, father_nid), paths))

        """3 node -> instances"""
        start_instance_set, end_instance_set = sp_online_utils.get_s_o_instances(paths)
        if son_nid in nodenid_to_instances_dict:
            new_set = nodenid_to_instances_dict[son_nid].intersection(start_instance_set)
            nodenid_to_instances_dict[son_nid] = new_set
        else:
            nodenid_to_instances_dict[son_nid] = start_instance_set
        if father_nid in nodenid_to_instances_dict:
            new_set = nodenid_to_instances_dict[father_nid].intersection(end_instance_set)
            nodenid_to_instances_dict[father_nid] = new_set
        else:
            nodenid_to_instances_dict[father_nid] = end_instance_set

    """print"""
    for node_nid, instances in nodenid_to_instances_dict.items():
        print ('#nodes alignment:\t', node_nid, len(instances)) #instances
    for (son_nid, father_nid), paths in son_father_to_paths_list:
        print('#all edge\'s paths:\t', son_nid, father_nid, len(paths))
        for path in paths:
            print ('#\t', path)

    """construct graph"""
    construct_grounded_graph = ConstructGroundedGraph(
        graph=graph,
        start_vertex=start_vertex,
        grounded_nodes=grounded_nodes,
        nodenid_to_instances_dict=nodenid_to_instances_dict,
        son_father_to_paths_list=son_father_to_paths_list,
        mediator_num=mediator_num,
        sequence_upper=sequence_upper)
    construct_grounded_graph.all_edge_sequence()
    return construct_grounded_graph.get_grounded_graph()


def donot_insert_node(son_node=None, father_node=None, nodenid_to_instances_dict=None):
    '''
    :param son_node:
    :param father_node:
    :return: one hop paths
    '''
    paths = set()
    if son_node.node_type == 'entity' and father_node.node_type == 'class':
        _, p_o = sp_candidate_generation_kb.get_s_p_p_o_bylinkedentity(son_node.id)
        if sp_online_utils.is_question_node(father_node):
            for p_o_ in p_o:
                paths.add("\t".join([son_node.id, p_o_]))
        else:
            paths = sp_online_utils.filters_by_constraits(subject_id=son_node.id, p_o_set=p_o, edge_friendly_name=None, system_o_classes_list=None)
    elif son_node.node_type == 'class' and father_node.node_type == 'class':
        if son_node.nid in nodenid_to_instances_dict:
            instances = nodenid_to_instances_dict[son_node.nid]
            for instance in instances:
                _, p_o = sp_candidate_generation_kb.get_s_p_p_o_bylinkedentity(instance)
                for p_o_ in p_o:
                    paths.add("\t".join([instance, p_o_]))
    elif son_node.node_type == 'literal' and father_node.node_type == 'class':
        # literal_value = '"'+son_node.id+'"' #356.72
        literal_value = literal_postprocess(son_node.id, q_mode=q_mode)
        s_p = sp_candidate_generation_kb.get_s_p_by_literal_none(literal_value=literal_value)
        for s_p_ in s_p:
            cols = s_p_.split('\t')
            path = "\t".join([son_node.id, cols[1], cols[0]])
            #print ('###literal path:\t', path)
            paths.add(path)
    else:
        pass
    return paths


def insert_one_node(son_node, father_node, nodenid_to_instances_dict=None):
    '''
    :param son_node:
    :param father_node:
    :return: add mediator node, paths list
    '''
    paths = set()
    if son_node.node_type == 'entity' and father_node.node_type == 'class':
        p1_m_p2_o = sp_candidate_generation_kb.get_p1_mediator_p2_answer(son_node.id)
        for p1_m_p2_o_ in p1_m_p2_o:
            p1 = p1_m_p2_o_.split('\t')[0]
            p2 = p1_m_p2_o_.split('\t')[2]
            # mediator_edge_reverse and mediator_edge
            if sp_online_utils.is_mediator_property_reverse_from_schema(p1) and sp_online_utils.is_mediator_property_from_schema(p2):
                paths.add("\t".join([son_node.id, p1_m_p2_o_]))
    elif son_node.node_type == 'class' and father_node.node_type == 'class':
        if son_node.nid in nodenid_to_instances_dict:
            instances = nodenid_to_instances_dict[son_node.nid]
            for instance in instances:
                p1_m_p2_o = sp_candidate_generation_kb.get_p1_mediator_p2_answer(instance)
                for p1_m_p2_o_ in p1_m_p2_o:
                    paths.add("\t".join([instance, p1_m_p2_o_]))
    elif son_node.node_type == 'literal' and father_node.node_type == 'class':
        # literal_value = '"'+son_node.id+'"' #356.72
        literal_value = literal_postprocess(son_node.id, q_mode=q_mode)
        s_p = sp_candidate_generation_kb.get_s_p_by_literal_none(literal_value=literal_value)
        for s_p_ in s_p:
            cols = s_p_.split('\t')
            p1 = cols[1]
            o1 = cols[0] #real s1
            if not sp_online_utils.is_mediator_property_from_schema(p1):
                continue
            s2_p2, _ = sp_candidate_generation_kb.get_s_p_p_o_bylinkedentity(o1)
            # for p2_o2_ in p2_o2:
            #     path = "\t".join([son_node.id, p1, o1, p2_o2_])
            #     print('###literal mediator path:\t', path)
            #     paths.add(path)
            for s2_p2_ in s2_p2:
                cols = s2_p2_.split('\t')
                path = "\t".join([son_node.id, p1, o1, cols[1], cols[0]]) # reverse direction
                print('###literal mediator path:\t', path)
                paths.add(path)
    else:
        pass
    return paths


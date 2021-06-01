from common_structs.ungrounded_graph import UngroundedEdge, UngroundedGraph, UngroundedNode
import copy
from method_sp.parsing import parsing_utils
from common_structs.graph import Digragh
from common_structs.cycle import DirectedCycle


def _generate_ungrounded_graph(nodes, edges, ungrounded_query_id=1):
    '''ungrounded graph'''
    return UngroundedGraph(
        ungrounded_query_id=ungrounded_query_id, nodes=nodes, edges=edges, important_words_list=[],
        abstract_question=[], grounded_linking=[], grounded_graph_forest=[])


def _generate_ungrounded_node(node_type, nid, friendly_name='blank_node', start_position=0, end_position=0, score=1.0):
    '''ungrounded node'''
    ungrounded_node = UngroundedNode(
        nid=nid, node_type=node_type,
        friendly_name=friendly_name,
        question_node=0, score=score,
        start_position=start_position, end_position=end_position)
    return ungrounded_node


def _generate_ungrounded_edge(start=-1, end=-1, friendly_name='blank_edge', score=1.0):
    '''ungrounded edge'''
    ungrounded_edge = UngroundedEdge(start=start, end=end, friendly_name=friendly_name, score=score)
    return ungrounded_edge


def update_ungrounded_graph_merge_question_node(ungrounded_graph):
    '''
    第5种改变: class{value:wh-word; question_node:1} -> class  转变为class的question_node:1
    :param ungrounded_graph:
    :return: ungrounded_graph
    '''
    from method_sp.parsing import node_recognition_utils
    ungrounded_graph = copy.deepcopy(ungrounded_graph)
    ungrounded_graph_nodes = ungrounded_graph.nodes
    merge_edges = []
    for edge in ungrounded_graph.edges:
        #question node=1; class
        class_question_node, class_node = parsing_utils.class_question_node_class_node_in_one_edge(ungrounded_graph_nodes, edge)
        if class_question_node is not None and class_node is not None:
            # friendly_name in wh-words
            is_equal_wh_word = node_recognition_utils.is_equal_wh_word(class_question_node.friendly_name)
            #chu du == 1
            search_adjacent_edges = parsing_utils.search_adjacent_edges(
                node=class_question_node, ungrounded_graph=ungrounded_graph)
            if is_equal_wh_word and len(search_adjacent_edges) == 1 and (edge.friendly_name == '' or edge.friendly_name == 'name'):
                # 满足上述三种条件以后，才merge
                merge_edges.append(edge)
    if len(merge_edges) == 0:
        return None
    new_ungrounded_edges = []
    new_ungrounded_nodes = []
    for edge in ungrounded_graph.edges:
        if edge in merge_edges:
            # question node=1; class
            class_question_node, class_node = parsing_utils.class_question_node_class_node_in_one_edge(ungrounded_graph_nodes, edge)
            #first check if it exist
            #if exist, update node information
            #if not exist, add node
            if parsing_utils.is_exist_in_nodes(new_ungrounded_nodes, class_node):
                temp_node = parsing_utils.search_one_node_in_nodes(new_ungrounded_nodes, class_node)
                temp_node.question_node = class_question_node.question_node
            else:
                class_node.question_node = class_question_node.question_node
                new_ungrounded_nodes.append(copy.deepcopy(class_node))
        else:
            start_node = parsing_utils.search_one_node_in_nodes_by_nid(ungrounded_graph_nodes, edge.start)
            end_node = parsing_utils.search_one_node_in_nodes_by_nid(ungrounded_graph_nodes, edge.end)
            #first check if it exist
            #if exist, no add the node
            #if not exist, add node
            if not parsing_utils.is_exist_in_nodes(new_ungrounded_nodes, start_node):
                new_ungrounded_nodes.append(start_node)
            if not parsing_utils.is_exist_in_nodes(new_ungrounded_nodes, end_node):
                new_ungrounded_nodes.append(end_node)
            new_ungrounded_edges.append(copy.deepcopy(edge))
    return _generate_ungrounded_graph(new_ungrounded_nodes, new_ungrounded_edges,ungrounded_query_id=ungrounded_graph.ungrounded_query_id+1)


def undate_ungrounded_graph_del_cycle(ungrounded_graph):
    '''破圈操作:  包含e-e或e-l或l-l的圈，要把它们破开。有圈情况:  event型问句,
    比如what were the compositions made by bach in 1749; O并列;  VP 并列; 修饰疑问短语，挂到了动词身上'''
    ungrounded_graph_edges = ungrounded_graph.edges
    di_graph = Digragh()
    for edge in ungrounded_graph_edges:
        di_graph.add_edge(edge.start, edge.end)
        di_graph.add_edge(edge.end, edge.start)
    directed_cycle = DirectedCycle(di_graph)
    if len(directed_cycle.all_cycles) > 0:
        return parsing_utils.del_edge_in_ungrounded_edge(ungrounded_graph, directed_cycle)
    else:
        return None


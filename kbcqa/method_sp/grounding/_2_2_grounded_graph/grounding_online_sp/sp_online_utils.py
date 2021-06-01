from common_structs.graph import Graph
from common_structs.grounded_graph import GroundedNode, GroundedEdge
from method_sp.grounding import grounding_args


def has_literal_node(nodes):
    '''literal node'''
    has_literal = False
    for node in nodes:
        if node.node_type == 'literal':
            has_literal = True
    return has_literal


def is_question_node(ungrounded_node):
    '''check if node is class node'''
    class_list = ["class"]
    if ungrounded_node.node_type in class_list and ungrounded_node.question_node == 1:
        return True
    else:
        return False


def search_question_node_nid(nodes):
    '''question node index'''
    question_node_index = None
    for node in nodes:
        if is_question_node(node):
            question_node_index = node.nid
    return question_node_index


def convert_triples_to_graph(grounded_edges):
    '''convert triples to graph'''
    g = Graph()
    for edge in grounded_edges:
        g.add_edge(edge.start, edge.end)
    return g


def search_one_node_in_nodes_by_nid(ungrounded_graph_nodes, nid):
    result = None
    for node in ungrounded_graph_nodes:
        if node.nid == nid:
            result = node
            break
    return result


def get_edge_by_nodes(edges, node_a, node_b):
    result = None
    for edge in edges:
        if (edge.start == node_a.nid and edge.end == node_b.nid) \
            or (edge.start == node_b.nid and edge.end == node_a.nid):
            result = edge
    return result


def add_path(_grounded_nodes, grounded_graph, path, son_id=None, father_id=None):
    '''add path in current grounded graph'''
    basic_son_node = search_one_node_in_nodes_by_nid(_grounded_nodes, son_id)
    basic_father_node = search_one_node_in_nodes_by_nid(_grounded_nodes, father_id)
    spo_list = path.split('\t')
    son_node = GroundedNode(nid=son_id, id=spo_list[0], score=1.0,node_type=basic_son_node.node_type, question_node=basic_son_node.question_node)
    father_node = GroundedNode(nid=father_id, id=spo_list[len(spo_list) - 1], score=1.0, node_type=basic_father_node.node_type, question_node=basic_father_node.question_node)
    grounded_graph.add_node(son_node)
    grounded_graph.add_node(father_node)
    if len(spo_list) > 3:  # s->{p1}->mediator->{p2}->o mediator or cvt
        middle_id = son_id * 10 + father_id
        middle_node = GroundedNode(nid=middle_id, id=spo_list[2], score=1, node_type='class', question_node=0)
        grounded_graph.add_node(middle_node)
        edge_1 = GroundedEdge(start=son_id, end=middle_id, relation=spo_list[1], friendly_name=spo_list[1], score=1.0)
        edge_2 = GroundedEdge(start=middle_id, end=father_id, relation=spo_list[3], friendly_name=spo_list[3], score=1.0)
        grounded_graph.add_edge(edge_1)
        grounded_graph.add_edge(edge_2)
    else:
        edge = GroundedEdge(start=son_id, end=father_id, relation=spo_list[1], friendly_name=spo_list[1], score=1.0)
        grounded_graph.add_edge(edge)


def dfs(_grounded_nodes, grounded_graph, marked, graph, vertex, edge_to_path_dict):
    marked[vertex] = True
    for w in graph.get_adjacent_vertices(vertex):
        if not marked[w]:
            path = edge_to_path_dict[(w, vertex)]
            add_path(_grounded_nodes, grounded_graph, path, son_id=w, father_id=vertex)
            dfs(_grounded_nodes, grounded_graph, marked, graph, w, edge_to_path_dict)


def get_reverse_property_from_lexcion(property, property_reverse_dict):
    '''get reverse property'''
    reverse_property = ''
    if property in property_reverse_dict.keys():
        reverse_property = property_reverse_dict[property][0]
    for key, value in property_reverse_dict.items():
        if key == property:
            reverse_property = value[0]
        if value[0] == property:
            reverse_property = key
    return reverse_property


def has_reverse_property(path_sequence):
    '''maybe mediator edge'''
    has_reverse_property_paths_pair = False
    for path_a in path_sequence:
        property_a = path_a.split('\t')[1]
        property_a_reverse = get_reverse_property_from_lexcion(property_a, grounding_args.property_reverse_dict)
        for path_b in path_sequence:
            if path_a == path_b:
                continue
            property_b = path_b.split('\t')[1]
            if property_b == property_a_reverse:
                has_reverse_property_paths_pair = True
    return has_reverse_property_paths_pair


#utils
def get_s_o_instances(paths):
    start_instance_set = set()
    end_instance_set = set()
    for path in paths:
        spo = path.split('\t')
        start_instance_set.add(spo[0])
        end_instance_set.add(spo[len(spo)-1]) #maybe spo's length is 5
    return start_instance_set, end_instance_set


def is_mediator_property_from_schema(property, schema_lines_list=None):
    result = False
    if schema_lines_list is None:
        schema_lines_list = grounding_args.schema_lines_list
    for schema_line in schema_lines_list:
        cols = schema_line.split('\t')
        if property == cols[2] and 'mediator' == cols[1]:
            result = True
            break
    return result


def is_mediator_property_reverse_from_schema(property):
    '''property is a reverse mediator edge'''
    reverse_property = get_reverse_property_from_lexcion(property, grounding_args.property_reverse_dict)
    is_reverse_mediator = is_mediator_property_from_schema(reverse_property, grounding_args.schema_lines_list)
    return is_reverse_mediator


def filters_by_constraits(subject_id, p_o_set, edge_friendly_name, system_o_classes_list, thresold=0.2):
    '''
    :param subject:
    :param p_o_set: all p_o paths
    :return: new p_o_set by filter
    '''
    new_s_p_o_set =set()
    for p_o_ in p_o_set:
        # po = p_o_.split('\t')
        new_s_p_o_set.add("\t".join([subject_id, p_o_]))
    return new_s_p_o_set


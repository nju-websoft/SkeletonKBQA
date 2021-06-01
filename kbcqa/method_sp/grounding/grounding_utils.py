from common_structs.graph import Graph
from common_structs.grounded_graph import GrounedGraph
from common_structs.depth_first_paths import DepthFirstPaths
from common.hand_files import read_list_yuanshi


def literal_postprocess(literal_value, q_mode):
    assert q_mode in ['graphq', 'cwq']
    if q_mode == 'graphq':
        if 'http://www.w3.org/2001/XMLSchema#datetime' in literal_value:
            literal_value = '"' + literal_value.replace('^^', '"^^<') + '>'
        elif '^^http://www.w3.org/2001/XMLSchema#double' in literal_value:
            literal_value = '"' + literal_value.replace('^^', '"^^<') + '>'
        elif 'http://www.w3.org/2001/XMLSchema#int' in literal_value:
            literal_value = '"' + literal_value.replace('^^', '"^^<') + '>'

    elif q_mode == 'cwq':
        # rule 1: lower()
        literal_value = literal_value.lower()
        # rule 2: date time
        if 'http://www.w3.org/2001/xmlschema#datetime' in literal_value:
            literal_value = literal_value.replace('^^http://www.w3.org/2001/xmlschema#datetime', "")
            literal_value = '"' + literal_value + '"^^xsd:dateTime'
        elif ',' in literal_value:  # rule 3: big int
            literal_value = literal_value.replace(',', '')
            literal_value = '"' + literal_value + '"'
        else:  # rule 4: string
            literal_value = '"' + literal_value + '"'
    return literal_value


def is_question_node(ungrounded_node):
    '''check if node is class node'''
    class_list = ["class"]
    if ungrounded_node.node_type in class_list and ungrounded_node.question_node == 1:
        return True
    else:
        return False


def get_question_node(nodes):
    result = None
    for node in nodes:
        if is_question_node(node):
            result = node
            break
    return result


def analysis_structure_category(_2_1_graph):
    '''2.1 query graph structure'''
    g = Graph()
    for edge in _2_1_graph.edges:
        g.add_edge(edge.start, edge.end)
    question_node = get_question_node(_2_1_graph.nodes)
    if question_node is None:
        #return None, None
        question_node = _2_1_graph.nodes[0]
    dfp = DepthFirstPaths(g, question_node.nid)
    path_list = []
    count_entities = 0
    for node in _2_1_graph.nodes:
        if node.node_type in ['entity', 'literal']:
            count_entities += 1
        if node.nid == question_node.nid:
            continue
        if _2_1_graph.get_node_degree(node) > 1:
            continue
        if dfp.has_path_to(node.nid):
            path_to_list = [i for i in dfp.path_to(node.nid)]
            path_list.append(path_to_list)

    category = 'other'  # composition, conjunction
    if len(path_list) == 1:
        if len(path_list[0]) == 2 and count_entities == 1:
            category = "composition-0" #[[1, 2]]
        elif len(path_list[0]) == 2 and count_entities == 2:
            category = "composition-2" #yes-no question
        else:
            category = "composition-1" #[[1, 2, 3, 4]]
    elif len(path_list) == 2:
        # category = "conjunction"
        if len(path_list[0]) == 2 and len(path_list[1]) == 2:
            category = 'conjunction-0'
        else:
            category = 'conjunction-1'
    return category


def candidate_query_to_grounded_graph(candidate_graphquerys):
    result = []
    for candidate_graphquery in candidate_graphquerys:
        result.append(
            GrounedGraph(type=candidate_graphquery["querytype"],
                         nodes=candidate_graphquery["nodes"],
                         edges=candidate_graphquery["edges"],
                         key_path=candidate_graphquery["path"],
                         denotation=candidate_graphquery['denotation']))
    return result


def read_literal_to_id_map(file_root):
    literal_to_id_dict = dict()
    lines = read_list_yuanshi(file_root+'literal_index.txt')
    for i, line in enumerate(lines): #1	"1995-04-07"^^<http://www.w3.org/2001/XMLSchema#datetime>
        cols = line.split('\t')
        literal_to_id_dict[cols[1]] = cols[0]
    return literal_to_id_dict


def get_local_filename(entities_or_literals):
    '''get file name'''
    filename_1 = None
    filename_2 = None
    if len(entities_or_literals) == 1:
        filename_1 = entities_or_literals[0][1] + '_' + entities_or_literals[0][0]
    elif len(entities_or_literals) == 2:
        filename_1 = entities_or_literals[0][1] + '_' + entities_or_literals[0][0]+'_' +entities_or_literals[1][1] + '_' + entities_or_literals[1][0]
        filename_2 = entities_or_literals[1][1] + '_' + entities_or_literals[1][0] +'_' + entities_or_literals[0][1] + '_' + entities_or_literals[0][0]
    return filename_1, filename_2

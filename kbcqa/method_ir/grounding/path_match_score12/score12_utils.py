from method_ir.grounding.path_match_score12 import triples_to_paths
import operator


def triples_to_paths_lcquad_e1e2(triples, entitys):
    goldpath = []
    for triple in triples:
        if triple['subject'] in entitys and triple['object'] in entitys:
            if triple['subject'] == entitys[0]:
                goldpath.append('+')
            else:
                goldpath.append('-')
        elif triple['subject'] == '?uri':
            goldpath.append('-')
        elif triple['subject'] == '?x' and triple['object'] == '?uri':
            goldpath.append('+')
        elif triple['subject'] == '?x' and triple['object'] != '?uri':
            goldpath.append('-')
        else:
            goldpath.append('+')
        goldpath.append(triple['predicate'])
    return goldpath


def _search_one_node_in_nodes_by_nid(ungrounded_graph_nodes, nid):
    result = None
    for node in ungrounded_graph_nodes:
        if node.nid == nid:
            result = node
            break
    return result


def get_triples_by_grounded_graph_edges_graphq(nodes, edges):
    triples = []
    for edge in edges:
        start_node = _search_one_node_in_nodes_by_nid(nodes, edge.start)
        end_node = _search_one_node_in_nodes_by_nid(nodes, edge.end)

        if start_node.node_type == 'entity' or start_node.node_type == 'literal':
            start_node_id = start_node.id
        elif start_node.node_type == 'class' and start_node.question_node == 1:
            start_node_id = '?x'
        else:
            start_node_id = '?'+start_node.id

        if end_node.node_type == 'entity' or end_node.node_type == 'literal':
            end_node_id = end_node.id
        elif end_node.node_type == 'class' and end_node.question_node == 1:
            end_node_id = '?x'
        else:
            end_node_id = '?'+end_node.id

        triple = dict()
        triple['subject'] = start_node_id
        triple['predicate'] = edge.relation
        triple['object'] = end_node_id
        triples.append(triple)
    return triples


def get_triples_by_grounded_graph_edges(nodes, edges):
    triples = []
    for edge in edges:
        start_node = _search_one_node_in_nodes_by_nid(nodes, edge.start)
        end_node = _search_one_node_in_nodes_by_nid(nodes, edge.end)
        triple = dict()
        triple['subject'] = start_node.id
        triple['predicate'] = edge.relation
        triple['object'] = end_node.id
        triples.append(triple)
    return triples


def grounded_graph_list_to_path_list(grounded_graph_forest):
    hop1 = []
    hop2 = []
    hop3 = []
    hop4 = []
    for grounded_graph in grounded_graph_forest:
        triples = get_triples_by_grounded_graph_edges(nodes=grounded_graph.nodes, edges=grounded_graph.edges)
        has_uri_answer_node = False
        for triple in triples:
            if triple['subject'] == '?a' or triple['object'] == '?a':
                has_uri_answer_node = True
        if has_uri_answer_node:
            paths = triples_to_path_list(triples=triples, _root_id='?a')

        else: #e1-to-e2
            entitys = []
            for triple in triples:
                if 'http://dbpedia.org/resource/' in triple['subject']: entitys.append(triple['subject'])
                if 'http://dbpedia.org/resource/' in triple['object']: entitys.append(triple['object'])
            entitys.sort()
            new_triples = rerank_triples(triples=triples)
            paths = triples_to_paths_lcquad_e1e2(triples=new_triples, entitys=entitys)

        if len(paths) == 1 * 2:
            hop1.append(paths)
        elif len(paths) == 2 * 2:
            hop2.append(paths)
        elif len(paths) == 3 * 2:
            hop3.append(paths)
        elif len(paths) == 4 * 2:
            hop4.append(paths)
    return hop1, hop2, hop3, hop4


def triples_to_path_list(triples, _root_id):
    def _convert_model_paths(path1):
        path1_predicate_list = []
        for i, path1_element in enumerate(path1):
            if i % 2:
                if path1_element.startswith('-'):
                    path1_predicate_list.append('-')
                    path1_predicate_list.append(path1_element[1:])
                else:
                    path1_predicate_list.append('+')
                    path1_predicate_list.append(path1_element)
        return path1_predicate_list
    return _convert_model_paths(triples_to_paths.convert_triple_to_paths(triples=triples, _root_id=_root_id))


def eq_paths(path1, path2):
    return operator.eq(path1, path2)


def evaluate_generation(graphq_candidates,goldorpred):
    accuracy=0
    for one in graphq_candidates:
        if 'path' not in one['gold']: continue
        goldpath=one['gold']['path']
        hops=[]
        for label in ['hop1','hop2','hop3','hop4']:  #['hop1','hop2','hop3','hop3_0','hop3_1','hop3_2', 'hop4']
            if label in one[goldorpred]:
                hops += one[goldorpred][label]
        match=False
        for hop in hops:
            if eq_paths(goldpath,hop):
                match=True
                break
        if match:
            accuracy += 1
            print(('%s\t%s') % (one['qid'], 1))
        else:
            print(('%s\t%s') % (one['qid'], 0))
    print(accuracy,len(graphq_candidates),accuracy/len( graphq_candidates))


def is_uri(_string):
    if _string.startswith('http://dbpedia.org/'):
        return True
    elif _string.startswith('http://rdf.freebase.com/ns'):
        return True
    elif not _string.startswith('?'):
        return True
    return False


def rerank_triples(triples):
    '''对triples集合重新排序
    优先顺序e > l > c > a     当有相同类型时候, 考虑字母顺序
    '''
    def _get_triple_by_entity_node(triples, node):
        triple = None
        for temp in triples:
            if node == temp['subject'] or node == temp['object']:
                triple = temp
        return triple

    def _get_triple_by_subject_class(triples, subject_class):
        triple = None
        for temp in triples:
            if subject_class == temp['subject'] and not is_uri(temp['object']):
                triple = temp
        return triple

    new_triples = []
    node_list = []
    for triple in triples:
        if is_uri(triple['subject']):
            node_list.append(triple['subject'])
        elif is_uri(triple['object']):
            node_list.append(triple['object'])
        else:
            node_list.append(triple['subject'])
    # sorted
    node_list = sorted(node_list)
    # add entity
    for node in node_list:
        if is_uri(node):
            new_triples.append(_get_triple_by_entity_node(triples=triples, node=node))
    # add class
    for node in node_list:
        if not is_uri(node):
            new_triples.append(_get_triple_by_subject_class(triples=triples, subject_class=node))
    return new_triples

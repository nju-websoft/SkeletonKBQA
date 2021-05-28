from skeleton_parsing import skeleton_parser_str
from itertools import chain
import networkx as nx


def get_deppath_list(question_normal, ungrounded_nodes, isSkeletonorDep='Skeleton'):
    abstract_question_deppath_list = []
    final_ph_tok_list, link_anchor_list, ans_anchor = placeholding_node(question_normal.split(' '), ungrounded_nodes=ungrounded_nodes)
    # print('#final_ph_tok_list:\t', final_ph_tok_list)
    # print('#link_anchor_list:\t', link_anchor_list)
    # print('#ans_anchor:\t', ans_anchor)
    try:
        path_tok_lists = dep_path_seq(ph_tok_list=final_ph_tok_list, link_anchor_list=link_anchor_list,
                                     ans_anchor=ans_anchor, isSkeletonorDep=isSkeletonorDep) #Skeleton  Dep
    except Exception as e:
        path_tok_lists = []
    for path_toks in path_tok_lists:
        token_path = ' '.join(path_toks)
        abstract_question_deppath_list.append(token_path.strip())
    return abstract_question_deppath_list


def triples(dep_parse, node=None):
    """
    Extract dependency triples of the form:
    ((head word, head tag), rel, (dep word, dep tag))
    """
    if not node:
        node = dep_parse.root
    head = node['address']
    for i in sorted(chain.from_iterable(node['deps'].values())):
        dep = dep_parse.get_by_address(i)
        yield (head, dep['rel'], dep['address'])
        for triple in triples(dep_parse=dep_parse, node=dep):
            yield triple


def get_redundancy_continuous_index(token_nodes, redundancy):
    '''get redundancy index of tokens
    phrase级的索引
    '''
    redundancy_tokens = redundancy.split(' ')
    j = 0
    start_index = -1
    end_index = -1
    while j < len(token_nodes):
        common = 0
        for redundancy_index in range(len(redundancy_tokens)):
            if redundancy_tokens[redundancy_index] == token_nodes[j]:
                common = common + 1
                j = j + 1
            else:
                j = j - common
                break
        if common == len(redundancy_tokens):
            start_index = j - common
            end_index = j - 1
        j = j + 1
    return start_index, end_index


def placeholding_node(tok_list, ungrounded_nodes):
    """ Step 1: Shrink each E/Tm link, occupying only one token """
    ph_tok_list = list(tok_list)
    link_pos_list = [-1] * len(tok_list)
    for link_idx, gl_data in enumerate(ungrounded_nodes):
        if gl_data.node_type != 'entity' and gl_data.node_type != 'literal':
            continue
        # st = gl_data.start_position  #start index
        # ed = gl_data.end_position  #end index
        st, ed = get_redundancy_continuous_index(token_nodes=tok_list, redundancy=gl_data.friendly_name)

        link_pos_list[ed] = link_idx
        # identifying the anchor word of the current linking  #[-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1]
        # for tok_idx in range(st, ed):
        #     ph_tok_list[tok_idx] = ''
            # print (ph_tok_list)   #['Who', 'is', 'the', 'office', 'holder', 'with', 'deputies', 'as', '', 'Brown', '?']

    tok_link_tups = []
    # 只保留根
    for ph_tok, link_idx in zip(ph_tok_list, link_pos_list):
        if ph_tok != '':
            tok_link_tups.append([ph_tok, link_idx])
            # remove non-trailing words of E/Tm  [['Who', -1], ['is', -1], ['the', -1], ['office', -1], ['holder', -1], ['with', -1], ['deputies', -1],
            # ['as', -1], ['Brown', 0], ['?', -1]]

    link_anchor_list = [-1] * len(ungrounded_nodes)  # the anchor position
    for anchor_idx, tup in enumerate(tok_link_tups):
        link_idx = tup[-1]
        if link_idx != -1:
            link_anchor_list[link_idx] = anchor_idx

    link_anchor_list_update = []
    for link_idx in link_anchor_list:
        if link_idx != -1:
            link_anchor_list_update.append(link_idx)
    link_anchor_list = link_anchor_list_update

    """ Step 2: Determine answer's anchor point """
    ans_anchor = 0        # find the first wh- word in the sentence, otherwise picking the first word
    # for tok_idx, (ph_tok, link_idx) in enumerate(tok_link_tups):
    #     if ph_tok.startswith('wh') or ph_tok == 'how' or ph_tok.startswith('Wh') or ph_tok == 'How':
    #         ans_anchor = tok_idx
    #         break
    for link_idx, gl_data in enumerate(ungrounded_nodes):
        if gl_data.question_node != 1:
            continue
        # st = gl_data.start_position  #start index
        ed = gl_data.end_position  #end index
        ans_anchor = ed

    """ Step 3: Dynamic replacement """
    # for anchor_idx in range(len(tok_link_tups)):
    #     ph_tok, link_idx = tok_link_tups[anchor_idx]
    #     if link_idx == -1:
    #         continue
        # tok_link_tups[anchor_idx][0] = 'XYZ'+str(link_idx)      # default value
    final_ph_tok_list = [tup[0] for tup in tok_link_tups]
    return final_ph_tok_list, link_anchor_list, ans_anchor


def dep_path_seq(ph_tok_list, link_anchor_list, ans_anchor, isSkeletonorDep='Skeleton'):
    placeholder_dict = {}
    for i, link_anchor in enumerate(link_anchor_list):
        placeholder_dict[link_anchor] = '<E'+str(i)+'>'

    utterance = ' '.join(ph_tok_list)
    assert isSkeletonorDep in ['Dep', 'Skeleton']
    if isSkeletonorDep == 'Dep':
        dependency_graph = skeleton_parser_str.get_dependency_tree(question=utterance)
    elif isSkeletonorDep == 'Skeleton':
        dependency_graph = skeleton_parser_str.get_hybrid_dependency_tree(question=utterance)

    edge_dict = {}
    for head_position, rel, dep_position in triples(dep_parse=dependency_graph):
        fwd_key = '%d-%d' % (head_position, dep_position)   #head -> depentant
        bkwd_key = '%d-%d' % (dep_position, head_position)  #depentant -> head
        edge_dict[fwd_key] = rel     #head -> depentant    ----> relation
        edge_dict[bkwd_key] = '!%s' % rel  #head -> depentant    ----> !relation

    path_tok_lists = []
    for _, link_anchor in enumerate(link_anchor_list):
        path_tok_list = find_path(ph_tok_list=ph_tok_list, dep_parse=dependency_graph, link_anchor=link_anchor,
                                  ans_anchor=ans_anchor, edge_dict=edge_dict, ph_dict=placeholder_dict)
        path_tok_lists.append(path_tok_list)
    return path_tok_lists


def find_path(ph_tok_list, dep_parse, link_anchor, ans_anchor, edge_dict, ph_dict):
    """
    :param dep_parse: dependency graph
    :param link_anchor: token index of the focus word (0-based)
    :param ans_anchor: token index of the answer (0-based)
    :param link_category: the category of the current focus link
    :param edge_dict: <head-dep, rel> dict
    :param ph_dict: <token_idx, ph> dict
    :return:
    """
    if ans_anchor != link_anchor:
        edges = []
        for head, rel, dep in triples(dep_parse=dep_parse):
            edges.append((head, dep))
        graph = nx.Graph(edges)
        path_nodes = nx.shortest_path(graph, source=ans_anchor+1, target=link_anchor+1) #[0, 1, 2, 3, 4]
    else:
        path_nodes = [link_anchor]
    path_tok_list = []
    path_len = len(path_nodes)
    if path_len > 0:
        for position in range(path_len-1):
            edge = edge_dict['%d-%d' % (path_nodes[position], path_nodes[position+1])]
            cur_token_idx = path_nodes[position] - 1
            if cur_token_idx in ph_dict:
                path_tok_list.append(ph_dict[cur_token_idx])
            else:
                path_tok_list.append(ph_tok_list[cur_token_idx])
            path_tok_list.append(edge)
        if link_anchor in ph_dict:
            path_tok_list.append(ph_dict[link_anchor])
        else:
            path_tok_list.append('<E>')
    return path_tok_list


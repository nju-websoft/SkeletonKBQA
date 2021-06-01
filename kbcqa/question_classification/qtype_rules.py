"""boolean questions"""
aux_set = {'is', 'did', 'are', 'do', 'was', 'does'}


"""count"""
count_phrases = ['Count', 'How many', 'how many', 'the number of', 'the count of', 'the amount of', 'total number of', 'count']
count_ner_tags = ['count']


"""graphq"""
# dayu_phrases = ['more', 'more than' ,'greater', 'higher', 'longer than', 'taller than']
dayu_dengyu_phrases = ['at least', 'not less than', 'or more']
# dengyu_phrases = ['equal', 'same']
# xiaoyu_phrases = ['earlier', 'less than', 'smaller', 'less', 'no higher than', 'fewer', 'fewer than']
xiaoyu_dengyu_phrases = ['at most', 'maximum', 'or less', 'no larger than']


"""cwq"""
xiaoyu_phrases = ['less', 'earlier than', 'before', 'prior', 'smaller than', 'fewer than', 'at least',
                  'lower than', 'under', 'below', 'less than', 'earlier', 'smaller','no higher than','fewer'] #'end'
dayu_phrases = ['above', 'exceeded', 'more than', 'larger than', 'after', 'later than', 'form',
                'larger than', 'higher than', 'greater than', 'over', 'from','more',
                'greater','higher','longer than','taller than']
comparative_ner_tags = ['>', '>=', '<', '<=']


argmin_phrases = ['smallest', 'least', 'weakest', 'minimum', 'minimal', 'youngest',
                  'closest', 'shortest', 'thinnest','tiniest','hollowest',
                  'narrowest','shallowest','simplest','latest','poorest','littlest','earliest','fewest','earliest?t','soonest','earliest-ending', 'earliest-founded',
                  'earlies','smalllest','lowest','earliest-opened','first','latest-starting']

argmax_phrases = ['largest', 'brightest', 'heaviest', 'most', 'maximum', 'maximal', 'ultimate', 'totally', 'hugest',
                  'longest', 'biggest', 'fattest', 'fastest',
                  'greatest', 'quickest', 'tallest', 'oldest', 'highest',
                  'eldest', 'heaviest', 'farthest', 'furthest', 'richest', 'best','lastest','newest','last']
arg_ner_tags = ['argmax', 'argmin']


""""""


def count_serialization(question):
    question_tokens_list = question.split(' ')
    serialization_list = ['O' for _ in question_tokens_list]
    for count_mention in count_phrases:
        if count_mention+' ' not in question:
            continue
        serialization_list = serialization_mention(question_tokens_list, count_mention.split(' '), ner_tag='count')
        break
    return serialization_list


def superlative_serialization(question):
    question_tokens_list = question.split(' ')
    serialization_list = ['O' for _ in question_tokens_list]
    for arg_mention in argmin_phrases:
        if arg_mention + ' ' not in question:
            continue
        serialization_list = serialization_mention(question_tokens_list, arg_mention.split(' '), ner_tag='argmin')
    for arg_mention in argmax_phrases:
        if arg_mention + ' ' not in question:
            continue
        serialization_list = serialization_mention(question_tokens_list, arg_mention.split(' '), ner_tag='argmax')
    return serialization_list


def comparative_serialization(question):
    question_tokens_list = question.split(' ')
    serialization_list = ['O' for _ in question_tokens_list]
    for mention in dayu_phrases:
        if mention not in question:
            continue
        serialization_list = serialization_mention(question_tokens_list, mention.split(' '), ner_tag='>')
        break
    for mention in dayu_dengyu_phrases:
        if mention not in question:
            continue
        serialization_list = serialization_mention(question_tokens_list, mention.split(' '), ner_tag='>=')
        break
    for mention in xiaoyu_phrases:
        if mention not in question:
            continue
        serialization_list = serialization_mention(question_tokens_list, mention.split(' '), ner_tag='<')
        break
    for mention in xiaoyu_dengyu_phrases:
        if mention not in question:
            continue
        serialization_list = serialization_mention(question_tokens_list, mention.split(' '), ner_tag='<=')
        break
    return serialization_list


""""""


def is_count_funct(serialization_list):
    is_count = False
    for element in serialization_list:
        if element in count_ner_tags:
            is_count = True
            break
    return is_count


def is_superlative_funct(serialization_list):
    is_superlative = False
    for element in serialization_list:
        if element in arg_ner_tags:
            is_superlative = True
            break
    return is_superlative


def is_comparative_funct(serialization_list):
    is_comarative = False
    for element in serialization_list:
        if element in comparative_ner_tags:
            is_comarative = True
            break
    return is_comarative


""""""


def serialization_mention(question_word_list, mention_word_list, is_lower=False, ner_tag='I'):
    '''
    input: question_string and entity_mention
    output: O O I I O O O O serialization
    example: 'find me tablet computers from apple inc.' and 'apple inc.'
    example output: O O O O O I I
    question = 'which presidents of the u.s. weighed 80.0 kilograms or more ?'
    class_mention = 'presidents of the u.s.'
    '''
    serialization_list = list()
    start_index = -1
    end_index = -1
    if is_lower:
        new_question_word_list = []
        for question_word in question_word_list:
            new_question_word_list.append(question_word.lower())
        question_word_list = new_question_word_list
        new_mention_word_list = []
        for mention_word in mention_word_list:
            new_mention_word_list.append(mention_word.lower())
        mention_word_list = new_mention_word_list
    for i in range(len(question_word_list)):
        for j in range(len(question_word_list)):
            if question_word_list[i:j] == mention_word_list:
                start_index = i
                end_index = j-1
    for i in range(len(question_word_list)):
        if start_index <= i <= end_index:
            serialization_list.append(ner_tag)
        else:
            serialization_list.append('O')
    return serialization_list


""""""


def get_superlative_type(question_normal):
    """What TV series was Mark Harmon the star of that ran the least amount of time on TV ?"""
    result = 'argmax'
    question_normal = question_normal.lower()
    superlative_serialization_list = superlative_serialization(question=question_normal)
    for element in superlative_serialization_list:
        if element in ['argmax', 'argmin']:
            result = element
            break
    return result


def get_comparative_type(question_normal):
    """what us presidents has a weight of at least 80.0 kg."""
    result = '>'
    question_normal = question_normal.lower()
    superlative_serialization_list = comparative_serialization(question=question_normal)
    for element in superlative_serialization_list:
        if element in ['>', '>=', '<', '<=']:
            result = element
            break
    return result


""""""


def set_class_aggregation_function(ungrounded_nodes=None, dependency_graph=None, surface_tokens=None):
    '''set ordinal function of class'''

    def is_comparative_by_token_ner_tag(token):
        result = False
        if token.ner_tag is not None and token.ner_tag in comparative_ner_tags:
            result = True
        return result

    def is_count_by_token_ner_tag(token):
        result = False
        if token.ner_tag is not None and token.ner_tag == 'count':
            result = True
        return result

    def is_superlative_by_token_ner_tag(token):
        result = False
        if token.ner_tag is not None and token.ner_tag in arg_ner_tags:
            result = True
        return result

    def search_for_node_by_index(node_index, dependency_graph):
        result_node = None
        for index, node in dependency_graph.nodes.items():
            if index == node_index:
                result_node = node
        return result_node

    def adj_edge_nodes_update(dep_node_index, dependency_graph):
        '''zhao children'''
        dep_result_nodes_index = []
        if dep_node_index is None:
            return dep_result_nodes_index
        dep_node = search_for_node_by_index(dep_node_index, dependency_graph)
        # chu du
        for _, child_index_list in dep_node['deps'].items():
            for child_index in child_index_list:
                dep_result_nodes_index.append(child_index)
        # ru du
        for node_index_, node in dependency_graph.nodes.items():
            for _, child_index_list in node['deps'].items():
                for child_index in child_index_list:
                    if child_index == dep_node_index:  # 某个顶点的孩子是dep_node, 那么他就是其父亲
                        dep_result_nodes_index.append(node_index_)
        return dep_result_nodes_index


    for ungrounded_node in ungrounded_nodes:
        # 只有class, literal上面设置聚合属性
        if ungrounded_node.node_type == 'entity':
            continue
        for surface_index in range(ungrounded_node.start_position, ungrounded_node.end_position+1):
            # 遍历node的每个word, 检测它的所有出边, node的索引+1以后, 就是依存中的索引
            adj_vertexs = adj_edge_nodes_update(surface_index + 1, dependency_graph)
            print('#####dep relations:', surface_index+1, adj_vertexs)
            for adj_vertex_index in adj_vertexs:
                adj_token = surface_tokens[adj_vertex_index-1]
                if is_count_by_token_ner_tag(adj_token):
                    ungrounded_node.function = 'count'
                elif is_superlative_by_token_ner_tag(adj_token):
                    ungrounded_node.function = adj_token.ner_tag
                elif is_comparative_by_token_ner_tag(adj_token):
                    ungrounded_node.function = adj_token.ner_tag
                else: # 再走一层
                    pass
                    # adj_adj_vertexs = parsing_utils.adj_edge_nodes_update(adj_vertex_index, dependency_graph)
                    # if adj_adj_vertexs is None:
                    #     continue
                    # for adj_adj_vertex_index in adj_adj_vertexs:
                    #     adj_adj_token = surface_tokens[adj_adj_vertex_index-1]
                    #     if counting.is_count_by_token_ner_tag(adj_adj_token):
                    #         ungrounded_node.function = 'count'
                    #     elif superlative.is_superlative_by_token_ner_tag(adj_adj_token):
                    #         ungrounded_node.function = adj_adj_token.ner_tag
                    #     elif comparative.is_comparative_by_token_ner_tag(adj_adj_token):
                    #         ungrounded_node.function = adj_adj_token.ner_tag
    return ungrounded_nodes


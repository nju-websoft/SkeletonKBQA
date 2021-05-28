

def update_headword_index(tokens, headword_index):
    result = headword_index
    for z_index, z_token in enumerate(tokens):
        if z_index == headword_index:
            result = z_token.index
            break
    return result


def look_for_position(redundancy_span, root_span_node):
    redundancy_tokens = redundancy_span.split(' ')
    j = 0
    start_index = -1
    end_index = -1
    while j < len(root_span_node.tokens):
        common = 0
        for redundancy_index in range(len(redundancy_tokens)):
            if redundancy_tokens[redundancy_index] == root_span_node.tokens[j].value:
                common = common + 1
                j = j + 1
            else:
                j = j - common
                break
        if common == len(redundancy_tokens):
            start_index = root_span_node.tokens[j - common].index
            end_index = root_span_node.tokens[j - 1].index
        j = j + 1
    if start_index == -1 and end_index == -1:
        j = 0
        while j < len(root_span_node.tokens):
            for redundancy_index in range(len(redundancy_tokens)):
                if redundancy_index == 0 and redundancy_tokens[redundancy_index] == root_span_node.tokens[j].value:
                    start_index = root_span_node.tokens[j].index
                if redundancy_index == len(redundancy_tokens) - 1 and redundancy_tokens[redundancy_index] == root_span_node.tokens[j].value:
                    end_index = root_span_node.tokens[j].index
            j = j + 1
    # print('#index:\t', start_index, end_index)
    if end_index < start_index:
        start_index = -1
        end_index = -1
    return start_index, end_index


def get_sub_tokens(tokens, start_index, end_index):
    sub_tokens = []
    for temp_token in tokens:
        if start_index <= temp_token.index <= end_index:
            sub_tokens.append(temp_token)
    return sub_tokens


def is_leaf(span_tree, span_node):
    '''判断是不是叶子'''
    is_leaf = True
    for id, node in span_tree.nodes.items():
        if node.isRoot: continue
        if node.headword_position is None: continue
        if span_node.start_position <= node.headword_position <= span_node.end_position:
            is_leaf = False
            break
    return is_leaf


def update_span_tree_structure(span_tree, sub_span_node):

    def _look_for_related_nodes(span_tree, span_node):
        '''找与span node相关联的span nodes'''
        related_nodes = []
        for id, node in span_tree.nodes.items():
            if node.isRoot: continue
            if node.headword_position is None: continue
            # if span_node.start_position <= node.headword_position <= span_node.end_position:
            for token in span_node.tokens:
                if token.index == node.headword_position:
                    related_nodes.append(node)
        return related_nodes

    related_span_nodes = _look_for_related_nodes(span_tree, sub_span_node)  # look for 相关联的顶点列表
    for related_span_node in related_span_nodes:
        yuanyou_father_span_node = span_tree.get_father_span_by_sonid(related_span_node.id)
        if yuanyou_father_span_node is not None:
            yuanyou_father_span_node.children.remove(related_span_node.id)
        span_tree.add_child_rel_with_headword(
            father_id=sub_span_node.id, son_id=related_span_node.id,
            headword_position=related_span_node.headword_position,
            headword_relation=related_span_node.headword_relation)


def update_span_tree_nodes(span_tree, start_index, end_index):
    # update root span node
    new_root_span_node_tokens = []
    for old_token in span_tree.tokens:
        if start_index <= old_token.index <= end_index:
            continue
        new_root_span_node_tokens.append(old_token)
    span_tree.set_tokens(new_root_span_node_tokens)


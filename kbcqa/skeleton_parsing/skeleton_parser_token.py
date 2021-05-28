from common_structs.skeleton import SpanTree
from skeleton_parsing.models_bert.fine_tuning_based_on_bert_interface import redundancy_span_interface
from skeleton_parsing.models_bert.fine_tuning_based_on_bert_interface import sequences_classifier_interface
from skeleton_parsing.models_bert.fine_tuning_based_on_bert_interface import headword_span_interface
from skeleton_parsing.models_bert.fine_tuning_based_on_bert_interface import simplif_classifier_interface
from skeleton_parsing import skeleton_utils


def span_tree_generation_only_dep(tokens):
    span_tree = SpanTree(tokens=tokens)
    span_tree.add_span_node(id=0, head_tail_position=[0, len(tokens)], isRoot=True, tokens=tokens)
    return span_tree


def span_tree_generation_head(tokens):
    '''
    产生叶子顶点
    产生非叶子顶点
    每个树的顶点, 视为tokens列表
    边: 视为顶点与另一顶点内的某个token之间关系.
    '''
    epoch = 0
    span_tree = SpanTree(tokens=tokens)
    root_span_node = span_tree.add_span_node(id=0, head_tail_position=[0, len(tokens)], isRoot=True, tokens=tokens)
    while simplif_classifier_interface.process(root_span_node.content) == 1:
        epoch = epoch + 1
        if epoch > 10:
            break

        """text span prediction"""
        redundancy_span = redundancy_span_interface.simple_process(root_span_node.content)
        if redundancy_span is None or redundancy_span == 'empty' or len(root_span_node.tokens) - len(redundancy_span.split(' ')) <= 3:
            """"heuristic rule, 如果删除以后, tokens数量小于4 超过10轮的迭代, 就退出"""
            break

        """head word"""
        headword_index = headword_span_interface.simple_process(question=root_span_node.content, span=redundancy_span)
        """update headword index, based on complete sequence"""
        headword_index = skeleton_utils.update_headword_index(tokens=root_span_node.tokens, headword_index=headword_index)

        """span position in the question"""
        start_index, end_index = skeleton_utils.look_for_position(redundancy_span, root_span_node)
        if start_index > end_index:
            break
        sub_tokens = skeleton_utils.get_sub_tokens(root_span_node.tokens, start_index=start_index, end_index=end_index)
        sub_span_node = span_tree.add_span_node(id=epoch, head_tail_position=[start_index, end_index], tokens=sub_tokens, isRoot=False)

        """增长树结构: 判断是叶子顶点还是非叶子顶点. span node部分是不是有其他node的根, 如果有, 则为非叶子顶点; 否则, 则为叶子顶点."""
        if not skeleton_utils.is_leaf(span_tree=span_tree, span_node=sub_span_node):
            """非叶子顶点, 等价于插入顶点"""
            skeleton_utils.update_span_tree_structure(span_tree=span_tree, sub_span_node=sub_span_node)

        """relation classifier"""
        relation = sequences_classifier_interface.process(line_a=root_span_node.content, line_b=redundancy_span)

        """add triple"""
        span_tree.add_child_rel_with_headword(father_id=root_span_node.id, son_id=sub_span_node.id, headword_position=headword_index, headword_relation=relation)

        """update question"""
        skeleton_utils.update_span_tree_nodes(span_tree=root_span_node, start_index=start_index, end_index=end_index)
    return span_tree


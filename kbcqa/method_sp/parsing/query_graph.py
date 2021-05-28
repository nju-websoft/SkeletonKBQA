from method_sp.parsing import parsing_utils, parsing_args, structure_transfers, node_recognition, relation_extraction_nff
from skeleton_parsing import fine_grained_dependency, skeleton_parser_token
from common_structs.question_annotation import QuestionAnnotation
from question_classification import qtype_rules, qtype_interface


def run_ungrounded_graph_interface(qid=None, question_normal=None, gold_graph_query=None, gold_answer=None, gold_sparql_query=None, node_is_gold=False, q_mode=None):
    ''' 1. span tree;  2. node mention annotation;   3. dependency tree;  4. relation extraction '''
    tokens = parsing_utils.create_tokens(question_normal.split(" "))
    if parsing_args.parser_mode == 'head':
        span_tree = skeleton_parser_token.span_tree_generation_head(tokens=tokens)
    else:
        span_tree = skeleton_parser_token.span_tree_generation_only_dep(tokens=tokens)
    print('#span tree:', span_tree)

    """3.generate dependency tree"""
    span_tree_hybrid_dependency_graph = fine_grained_dependency.span_tree_to_hybrid_dependency_graph_interface(span_tree=span_tree)
    span_tree_hybrid_dependency_graph = parsing_utils.update_dependencygraph_indexs(old_dependency_graph=span_tree_hybrid_dependency_graph)

    """node mention annotation sequence_bert_ner_tag_dict"""
    if node_is_gold:
        ungrounded_nodes = node_recognition.generate_gold_nodes(question_normal=question_normal)
    else:
        ungrounded_nodes = node_recognition.generate_nodes(question_normal=question_normal, qid=qid, tokens=tokens)

    # aggregation
    is_agg, serialization_list = qtype_interface.aggregation_interface(question_normal=question_normal)
    if is_agg != 'none':
        for i, token in enumerate(tokens):
            if serialization_list[i] != 'O' and token.ner_tag is None:
                token.ner_tag = serialization_list[i]
        qtype_rules.set_class_aggregation_function(ungrounded_nodes=ungrounded_nodes,
                                                  dependency_graph=span_tree_hybrid_dependency_graph, surface_tokens=tokens)

    # nff's ungrounded graph
    super_ungrounded_graph = relation_extraction_nff.generate_ungrounded_graph(ungrounded_nodes=ungrounded_nodes, span_tree_hybrid_dependency_graph=span_tree_hybrid_dependency_graph)

    ungrounded_graphs_list = []
    super_ungrounded_graph.set_blag('super')
    ungrounded_graphs_list.append(super_ungrounded_graph)

    """操作：折叠合并 疑问词节点连接class的话, 折叠疑问词, 连接的class的question node设为1
        class{value:wh-word; question_node:1} -> class  转变为class的question_node:1
    """
    merge_question_ungrouned_graph = structure_transfers.update_ungrounded_graph_merge_question_node(ungrounded_graph=super_ungrounded_graph)
    if merge_question_ungrouned_graph is not None:
        merge_question_ungrouned_graph.set_blag('merge_qc')
        ungrounded_graphs_list.append(merge_question_ungrouned_graph)

    """操作: 破圈操作:  包含e-e或e-l或l-l的圈，要把它们破开"""
    if merge_question_ungrouned_graph is not None:
        current_ungrounded_graph = merge_question_ungrouned_graph
    else:
        current_ungrounded_graph = super_ungrounded_graph
    del_cycle_ungrounded_graph = structure_transfers.undate_ungrounded_graph_del_cycle(ungrounded_graph=current_ungrounded_graph)
    if del_cycle_ungrounded_graph is not None:
        del_cycle_ungrounded_graph.set_blag('del_cycle')
        ungrounded_graphs_list.append(del_cycle_ungrounded_graph)

    abstract_question_word = parsing_utils.abstract_question_word_generation(tokens=tokens, ungrounded_nodes=ungrounded_nodes)
    sequence_bert_ner_tag_dict = parsing_utils.get_nertag_sequence(ungrounded_nodes=ungrounded_nodes)
    main_ungrounded_graph = None
    for z, ungrounded_graph in enumerate(ungrounded_graphs_list):
        ungrounded_graph.sequence_ner_tag_dict = str(sequence_bert_ner_tag_dict)
        ungrounded_graph.abstract_question = str(abstract_question_word)
        ungrounded_graph.important_words_list = str(parsing_utils.importantwords_by_unimportant_abstractq(abstract_question_word))
        if z == len(ungrounded_graphs_list) - 1:
            main_ungrounded_graph = ungrounded_graph

    compositionality_type, q_function = qtype_interface.question_type_interface(question_normal=question_normal, dataset=q_mode)
    question_annotation = QuestionAnnotation(qid=qid,
                                             question=question_normal,
                                             question_normal=question_normal,
                                             span_tree=span_tree,
                                             span_tree_hybrid_dependency_graph=span_tree_hybrid_dependency_graph,
                                             main_ungrounded_graph=main_ungrounded_graph,
                                             sequence_ner_tag_dict=sequence_bert_ner_tag_dict,
                                             gold_graph_query=gold_graph_query,
                                             gold_answer=gold_answer,
                                             gold_sparql_query=gold_sparql_query,
                                             compositionality_type=compositionality_type,
                                             q_function=q_function)

    structure = question_annotation.convert_to_structure()
    structure.set_ungrounded_graph_forest(ungrounded_graph_forest=ungrounded_graphs_list)
    return structure


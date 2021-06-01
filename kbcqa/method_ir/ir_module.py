import os

from common_structs.ungrounded_graph import UngroundedGraph
from common_structs.grounded_graph import GrounedGraph, GroundedNode
from common.hand_files import write_structure_file, read_structure_file
from common_structs.question_annotation import QuestionAnnotation
from method_sp.parsing import parsing_utils, node_recognition
from question_classification import qtype_interface


def _run_ner_topic_entities(qid=None, question_normal=None, gold_graph_query=None, gold_answer=None, gold_sparql_query=None,
                            node_is_gold=False, q_mode='lcquad'):
    """node mention annotation"""
    tokens = parsing_utils.create_tokens(question_normal.split(" "))
    """generate nodes"""
    if node_is_gold:
        ungrounded_nodes = node_recognition.generate_gold_nodes(question_normal=question_normal)
    else:
        ungrounded_nodes = node_recognition.generate_nodes(question_normal=question_normal, tokens=tokens)
    ungrounded_graphs_list = []
    main_ungrounded_graph = UngroundedGraph(
        ungrounded_query_id=1, nodes=ungrounded_nodes, edges=[],
        important_words_list=[], abstract_question=[], grounded_linking=[], grounded_graph_forest=[])
    main_ungrounded_graph.set_blag('ir')
    """"""
    abstract_question_word = parsing_utils.abstract_question_word_generation(tokens=tokens,
                                                                             ungrounded_nodes=ungrounded_nodes)
    sequence_bert_ner_tag_dict = parsing_utils.get_nertag_sequence(ungrounded_nodes=ungrounded_nodes)
    main_ungrounded_graph.sequence_ner_tag_dict = str(sequence_bert_ner_tag_dict)
    main_ungrounded_graph.abstract_question = str(abstract_question_word)
    main_ungrounded_graph.important_words_list = str([])
    ungrounded_graphs_list.append(main_ungrounded_graph)
    """question type"""
    compositionality_type, q_function = qtype_interface.question_type_interface(question_normal=question_normal, q_mode=q_mode)
    """generate question annotation"""
    question_annotation = QuestionAnnotation(qid=qid,
                                             question=question_normal,
                                             question_normal=question_normal,
                                             tokens=tokens,
                                             span_tree=None,
                                             span_tree_hybrid_dependency_graph=None,
                                             main_ungrounded_graph=main_ungrounded_graph,
                                             sequence_ner_tag_dict=sequence_bert_ner_tag_dict,
                                             gold_graph_query=gold_graph_query,
                                             gold_answer=gold_answer,
                                             gold_sparql_query=gold_sparql_query,
                                             compositionality_type=compositionality_type,
                                             q_function=q_function)
    """generate structure"""
    structure = question_annotation.convert_to_structure()
    structure.set_ungrounded_graph_forest(ungrounded_graph_forest=ungrounded_graphs_list)
    return structure


def _ungrounded_graph_to_grounded_graph(ungrounded_graph, grounding_result_list):
    grouned_graph_list = []
    if grounding_result_list is None:
        return grouned_graph_list
    grounded_nodes = []
    for ungrounded_node in ungrounded_graph.nodes:
        grounded_nodes.append(GroundedNode(nid=ungrounded_node.nid,
                                           node_type=ungrounded_node.node_type,
                                           type_class=ungrounded_node.type_class,
                                           friendly_name=ungrounded_node.friendly_name,
                                           question_node=ungrounded_node.question_node,
                                           function=ungrounded_node.function,
                                           score=0))
    for grounded_node in grounded_nodes:
        for ungrounded_node, nodes_grounding in grounding_result_list:
            if grounded_node.nid == ungrounded_node.nid:
                """nodes_grounding: {'en.xtracycle':1.6, 'freebase.type_profile':1.0}"""
                for mid, pro in nodes_grounding.items():
                    grounded_node.id = mid
                    grounded_node.score = pro
                    break
    grounded_graph = GrounedGraph(grounded_query_id=ungrounded_graph.ungrounded_query_id,
                                  type='', nodes=grounded_nodes, edges=[], key_path='',
                                  sparql_query='', score=0, denotation='')
    grouned_graph_list.append(grounded_graph)
    return grouned_graph_list


"""1. nerd for DBpedia"""


def run_topics_entity_generation_dbpedia(tuples_list, node_is_gold=False, linking_is_gold=False, q_mode='lcquad'):
    from datasets_interface.question_interface import lcquad_1_0_interface
    if not linking_is_gold:
        from method_sp.grounding._2_1_grounded_graph.node_linking import node_linking_interface_dbpedia
    structure_list = []
    error_qid_list = []
    for i, (qid, question_normal, gold_sparql_query, gold_answer) in enumerate(tuples_list):
        print(('%d\t%s') % (i, question_normal))
        try:
            """ner"""
            structure = _run_ner_topic_entities(qid=qid, question_normal=question_normal,
                                                gold_sparql_query=gold_sparql_query,
                                                gold_answer=gold_answer,
                                                node_is_gold=node_is_gold,
                                                q_mode=q_mode)
            """entity linking"""
            for index, ungrounded_graph in enumerate(structure.get_ungrounded_graph_forest()):
                if index != len(structure.get_ungrounded_graph_forest()) - 1:
                    continue
                grounding_result_list = []
                for node in ungrounded_graph.nodes:
                    if linking_is_gold:
                        result_dict = lcquad_1_0_interface.get_topic_entities_by_question_and_mention(question=structure.question, mention=node.friendly_name)
                        grounding_result_list.append((node, result_dict))
                    else:
                        grounding_result_list.append((node, node_linking_interface_dbpedia.node_linking(qid=structure.qid, node=node)))
                grouned_graph_list = _ungrounded_graph_to_grounded_graph(ungrounded_graph=ungrounded_graph, grounding_result_list=grounding_result_list)
                ungrounded_graph.set_grounded_linking(grounding_result_list)
                ungrounded_graph.set_grounded_graph_forest(grouned_graph_list)
            structure_list.append(structure)
        except Exception as e:
            print('#Error:', question_normal, e)
            error_qid_list.append(question_normal)
    print('Error:', error_qid_list)
    return structure_list


"""1. nerd for Freebase"""


def run_topics_entity_generation_freebase(tuples_list, node_is_gold=False, linking_is_gold=False, q_mode='graphq'):
    from datasets_interface.question_interface import graphquestion_interface
    from datasets_interface.question_interface import complexwebquestion_interface
    from common import globals_args
    from method_sp.grounding._2_1_grounded_graph.node_linking import node_linking_interface_freebase
    assert q_mode in ['graphq', 'cwq']
    if q_mode == 'cwq':  # aqqu entity linking
        from method_sp.grounding._2_1_grounded_graph.node_linking.entity_linking_aqqu_vocab.surface_index_memory import EntitySurfaceIndexMemory
        elp = EntitySurfaceIndexMemory(entity_list_file=globals_args.kb_freebase_latest_file.entity_list_file,
                                       surface_map_file=globals_args.kb_freebase_latest_file.surface_map_file,
                                       entity_index_prefix=globals_args.kb_freebase_latest_file.entity_index_prefix)
    elif q_mode == 'graphq':
        from method_sp.grounding._2_1_grounded_graph.node_linking.entity_linking_en_vocab.entity_link_pipeline import EntityLinkPipeline
        elp = EntityLinkPipeline(freebase_graph_name_entity_file=globals_args.kb_freebase_en_2013.freebase_graph_name_entity,
            freebase_graph_alias_entity_file=globals_args.kb_freebase_en_2013.freebase_graph_alias_entity,
            graphquestions_train_friendlyname_entity_file=globals_args.kb_freebase_en_2013.graphquestions_train_friendlyname_entity,
            clueweb_mention_pro_entity_file=globals_args.kb_freebase_en_2013.clueweb_mention_pro_entity)

    structure_list = []
    error_qid_list = []
    for i, (qid, question_normal, gold_sparql_query, gold_answer) in enumerate(tuples_list):
        print(('%d\t%s') % (i, question_normal))
        # try:
        """ner"""
        structure = _run_ner_topic_entities(qid=qid, question_normal=question_normal,
                                            gold_sparql_query=gold_sparql_query,
                                            gold_answer=gold_answer,
                                            node_is_gold=node_is_gold,
                                            q_mode=q_mode)
        """entity linking"""
        for index, ungrounded_graph in enumerate(structure.get_ungrounded_graph_forest()):
            if index != len(structure.get_ungrounded_graph_forest()) - 1:
                continue
            grounding_result_list = []
            for node in ungrounded_graph.nodes:
                if linking_is_gold:
                    result_dict = dict()
                    if q_mode == 'graphq':
                        result_dict = graphquestion_interface.get_topic_entities_by_question_and_mention(
                            question=structure.question, mention=node.friendly_name)
                    elif q_mode == 'cwq':
                        result_dict = complexwebquestion_interface.get_topic_entities_by_question_and_mention(
                            question_normal=structure.question, mention=node.friendly_name)
                    grounding_result_list.append((node, result_dict))
                else:
                    grounding_result_list.append((node, node_linking_interface_freebase.node_linking(node=node, top_k=10, elp=elp)))
            grouned_graph_list = _ungrounded_graph_to_grounded_graph(ungrounded_graph=ungrounded_graph, grounding_result_list=grounding_result_list)
            ungrounded_graph.set_grounded_linking(grounding_result_list)
            ungrounded_graph.set_grounded_graph_forest(grouned_graph_list)
        structure_list.append(structure)
        # except Exception as e:
        #     print('#Error:', question_normal, e)
        #     error_qid_list.append(question_normal)
    print('Error:', error_qid_list)
    return structure_list


"""2. generate candidate answers"""


def run_candidate_graph_generation(structure_with_1_ungrounded_lcquad_file, output_file, q_mode='lcquad'):
    from method_ir.grounding import graph_2_1_to_2_2_ir
    from method_sp.grounding import grounded_graph_to_sparql
    from method_sp.grounding import sparql_to_denotation
    import os
    structure_list = read_structure_file(structure_with_1_ungrounded_lcquad_file)
    error_qid_list = []
    for _, structure in enumerate(structure_list):
        if str(structure.qid) + '.json' in os.listdir(output_file):
            continue
        print(structure.qid)
        compositionality_type = structure.compositionality_type
        for j, ungrounded_graph in enumerate(structure.ungrounded_graph_forest):
            if j != len(structure.ungrounded_graph_forest) - 1:
                continue
            grounded_graph_forest = []
            for _2_1_grounded_graph in ungrounded_graph.get_grounded_graph_forest():
                try:
                    if q_mode == 'graphq':
                        grounded_graph_forest.extend(graph_2_1_to_2_2_ir.get_oracle_graphs_by_2_1_graph_graphq(_2_1_grounded_graph=_2_1_grounded_graph, qtype=compositionality_type))
                    elif q_mode == 'cwq':
                        grounded_graph_forest.extend(graph_2_1_to_2_2_ir.get_oracle_graphs_by_2_1_graph_cwq(_2_1_grounded_graph=_2_1_grounded_graph, qtype=compositionality_type))
                    elif q_mode == 'lcquad':
                        grounded_graph_forest.extend(graph_2_1_to_2_2_ir.get_oracle_graphs_by_2_1_graph_lcquad(_2_1_grounded_graph=_2_1_grounded_graph, qtype=compositionality_type))
                except Exception as e:
                    print('#Error:', structure.qid, e)
                    grounded_graph_forest.clear()
                    error_qid_list.append(structure.qid)
                break
            for z in range(len(grounded_graph_forest)):
                grounded_graph_forest[z].grounded_query_id = ungrounded_graph.ungrounded_query_id * 100000 + z
                grounded_graph_forest[z].sparql_query = grounded_graph_to_sparql.grounded_graph_to_sparql(grounded_graph=grounded_graph_forest[z],
                                                                                                          q_function=structure.function,
                                                                                                          q_compositionality_type=structure.compositionality_type,
                                                                                                          q_mode=q_mode)
                grounded_graph_forest[z].denotation = sparql_to_denotation.set_denotation(grounded_graph=grounded_graph_forest[z],
                                                                                          q_compositionality_type=structure.compositionality_type)
            ungrounded_graph.set_grounded_graph_forest(grounded_graph_forest)
            print('#size:\t', len(grounded_graph_forest))
            if len(grounded_graph_forest) > 0:
                write_structure_file([structure], output_file + str(structure.qid) + '.json')
    print('Error qid list:', error_qid_list)


"""3. semantic matching"""


def run_grounding_graph_score12_match(input_file_folder, q_mode='lcquad'):
    """path candidate grounding graph"""
    from method_ir.grounding.path_match_score12.path_match_interface import PathMatchScore12
    path_match_score12 = PathMatchScore12(q_mode)
    for path in os.listdir(input_file_folder):
        structure_with_grounded_graphq_file = input_file_folder + path
        print(path)
        structure_list = read_structure_file(structure_with_grounded_graphq_file)
        for structure in structure_list:
            question = structure.question
            for j, ungrounded_graph in enumerate(structure.ungrounded_graph_forest):
                if j != len(structure.ungrounded_graph_forest) - 1: continue
                grounded_graph_list = ungrounded_graph.get_grounded_graph_forest()
                try:
                    bert_scores = path_match_score12.set_bert_score_score12(question_normal=question, grounded_graph_forest_list=grounded_graph_list)
                    for grounded_graph, bert_score in zip(grounded_graph_list, bert_scores):
                        grounded_graph.score = bert_score
                except Exception as e:
                    for grounded_graph in grounded_graph_list:
                        grounded_graph.score = 0.0
                    print('error')
        write_structure_file(structure_list, structure_with_grounded_graphq_file)
    print('over')

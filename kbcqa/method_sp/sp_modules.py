from common.hand_files import write_structure_file, read_structure_file
import os

"""1. ungrounded query generation"""


def run_query_graph_generation(tuples_list, node_is_gold=False, parser_mode='skeleton', q_mode='lcquad'):
    from method_sp.parsing import query_graph
    structure_list = []
    error_qid_list = []
    for i, (qid, question_normal, gold_sparql_query, gold_answer) in enumerate(tuples_list):
        print(('%d\t%s') % (i, question_normal))
        try:
            structure = query_graph.run_ungrounded_graph_interface(qid=qid, question_normal=question_normal,
                                                                   gold_sparql_query=gold_sparql_query,
                                                                   gold_answer=gold_answer, node_is_gold=node_is_gold,
                                                                   q_mode=q_mode, parser_mode=parser_mode)  # gold_graph_query=gold_graph_query
            structure_list.append(structure)
        except Exception as e:
            print('#Error:', i, e)
            error_qid_list.append(i)
    print('Error:', error_qid_list)
    return structure_list


"""2. entity linking for DBpedia"""


def run_grounded_node_grounding_dbpedia(structure_with_ungrounded_graphq_file, output_file, linking_is_gold=False):
    '''
    function: 1.0 ungrounded query  ->  2.1 grounded query
    input: structure_ungrounded_graphq_file
    :return: grounded graph with entity linking
    '''
    from datasets_interface.question_interface import lcquad_1_0_interface
    from method_sp.grounding._2_1_grounded_graph.node_linking import node_linking_interface_dbpedia
    from method_sp.grounding._2_1_grounded_graph.grounded_graph_2_1_generation import generate_grounded_graph_interface
    structure_list = read_structure_file(structure_with_ungrounded_graphq_file)
    for structure in structure_list:
        print(structure.qid)
        for i, ungrounded_graph in enumerate(structure.get_ungrounded_graph_forest()):
            if i != len(structure.get_ungrounded_graph_forest()) - 1:
                continue
            grounding_result_list = []
            for node in ungrounded_graph.nodes:
                if linking_is_gold:
                    result_dict = lcquad_1_0_interface.get_topic_entities_by_question_and_mention(question=structure.question, mention=node.friendly_name)
                    grounding_result_list.append((node, result_dict))
                else:
                    grounding_result_list.append((node, node_linking_interface_dbpedia.node_linking(node=node)))
            grouned_graph_list = generate_grounded_graph_interface(ungrounded_graph=ungrounded_graph, grounding_result_list=grounding_result_list)
            ungrounded_graph.set_grounded_linking(grounding_result_list)
            ungrounded_graph.set_grounded_graph_forest(grouned_graph_list)
    write_structure_file(structure_list, output_file)


"""2. entity linking for Freebase"""


def run_grounded_node_grounding_freebase(structure_with_ungrounded_graphq_file, output_file, linking_is_gold=False, q_mode='graphq'):
    '''
    function: 1.0 ungrounded query  ->  2.1 grounded query
    input: structure_ungrounded_graphq_file
    :return: grounded graph with entity linking
    '''
    from common import globals_args
    from datasets_interface.question_interface import graphquestion_interface
    from datasets_interface.question_interface import complexwebquestion_interface
    from method_sp.grounding._2_1_grounded_graph.node_linking import node_linking_interface_freebase
    from method_sp.grounding._2_1_grounded_graph.grounded_graph_2_1_generation import generate_grounded_graph_interface
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

    structure_list = read_structure_file(structure_with_ungrounded_graphq_file)
    for structure in structure_list:
        print(structure.qid)
        for i, ungrounded_graph in enumerate(structure.get_ungrounded_graph_forest()):
            if i != len(structure.get_ungrounded_graph_forest()) - 1:
                continue
            grounding_result_list = []
            for node in ungrounded_graph.nodes:
                if linking_is_gold:
                    assert q_mode in ['graphq', 'cwq']
                    result_dict = dict()
                    if q_mode == 'graphq':
                        if node.node_type in ['entity', 'class', 'literal']:
                            result_dict = graphquestion_interface.get_topic_entities_by_question_and_mention(question=structure.question, mention=node.friendly_name)
                    elif q_mode == 'cwq':
                        if node.node_type in ['entity', 'class', 'literal']:
                            result_dict = complexwebquestion_interface.get_topic_entities_by_question_and_mention(question_normal=structure.question, mention=node.friendly_name)
                    grounding_result_list.append((node, result_dict))
                else:
                    grounding_result_list.append((node, node_linking_interface_freebase.node_linking(node=node, elp=elp)))
            grouned_graph_list = generate_grounded_graph_interface(ungrounded_graph=ungrounded_graph, grounding_result_list=grounding_result_list)
            ungrounded_graph.set_grounded_linking(grounding_result_list)
            ungrounded_graph.set_grounded_graph_forest(grouned_graph_list)
    write_structure_file(structure_list, output_file)


"""3. generate candidate grounded queries"""


def run_grounded_graph_generation_by_structure(structure_with_grounded_graphq_node_grounding_file, output_file, q_mode='lcquad'):
    from method_sp.grounding._2_2_grounded_graph import graph_2_1_to_2_2_sp
    from method_sp.grounding import grounded_graph_to_sparql
    from method_sp.grounding import sparql_to_denotation
    structure_list = read_structure_file(structure_with_grounded_graphq_node_grounding_file)
    error_qid_list = []
    for i, structure in enumerate(structure_list):
        if str(structure.qid) + '.json' in os.listdir(output_file):
            continue
        print(structure.qid)
        for j, ungrounded_graph in enumerate(structure.ungrounded_graph_forest):
            if j != len(structure.ungrounded_graph_forest) - 1:
                continue
            grounded_graph_forest = []
            for _2_1_grounded_graph in ungrounded_graph.get_grounded_graph_forest():
                try:
                    assert q_mode in ['lcquad', 'graphq', 'cwq']
                    if q_mode == 'cwq':
                        grounded_graph_forest.extend(graph_2_1_to_2_2_sp.get_oracle_graphs_by_2_1_graph_cwq(_2_1_grounded_graph=_2_1_grounded_graph, qtype=structure.compositionality_type))
                    elif q_mode == 'graphq':
                        grounded_graph_forest.extend(graph_2_1_to_2_2_sp.get_oracle_graphs_by_2_1_graph_graphq(_2_1_grounded_graph=_2_1_grounded_graph, qtype=structure.compositionality_type))
                    elif q_mode == 'lcquad':
                        grounded_graph_forest.extend(graph_2_1_to_2_2_sp.get_oracle_graphs_by_2_1_graph(_2_1_grounded_graph=_2_1_grounded_graph))
                except Exception as e:
                    print('#Error:', structure.qid, e)
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
            if len(grounded_graph_forest) > 0:
                write_structure_file([structure], output_file + str(structure.qid) + '.json')
    print('Error qid list:', error_qid_list)


"""3. semantic matching"""


def run_grounding_graph_score12_match(input_file_folder, q_mode='lcquad'):
    from method_ir import ir_module
    ir_module.run_grounding_graph_score12_match(input_file_folder=input_file_folder, q_mode=q_mode)

from common.hand_files import write_structure_file
from common import globals_args
from method_sp import sp_running_interface
from evaluation import kbcqa_evaluation

"""1 parsing"""


def run_ungrounded_graph_from_graphq(graph_questions_filepath, output_file, node_is_gold=False):
    from datasets_interface.question_interface import graphquestion_interface
    graph_questions_struct = graphquestion_interface.read_graph_question_json(graph_questions_filepath)
    tuples_list = []
    for i, graphquestion in enumerate(graph_questions_struct):
        tuples_list.append((graphquestion.qid, graphquestion.question, graphquestion.graph_query, graphquestion.answer))
    print(len(tuples_list))
    structure_list = sp_running_interface.run_query_graph_generation(tuples_list=tuples_list, node_is_gold=node_is_gold)
    write_structure_file(structure_list, output_file)


if __name__ == '__main__':
    module = "2.2"
    print('#module:', module)
    graph_questions_filepath = globals_args.fn_graph_file.graphquestions_testing_dir
    output_path = globals_args.fn_graph_file.dataset + 'output_graphq_e2e/output_graphq_sp_dep_slot_e2e_sp4_v0.1_wo_agg_完成'
    structure_with_1_ungrounded_graphq_file = output_path + '/structures_with_1_grounded_graph_skeleton_test_0130.json'
    structure_with_2_1_grounded_graph_file = output_path + '/structures_with_2_1_grounded_graph_skeleton_train_0130.json'
    structure_with_2_2_grounded_graph_folder = output_path + '/2.2_train_1509_no_agg/'

    # module
    if module == '1_ungrounded_query_generation':
        run_ungrounded_graph_from_graphq(graph_questions_filepath, structure_with_1_ungrounded_graphq_file, node_is_gold=False)
    elif module == '2.1_entity_linking':
        sp_running_interface.run_grounded_node_grounding_freebase(structure_with_1_ungrounded_graphq_file,
                                                                  structure_with_2_1_grounded_graph_file,
                                                                  linking_is_gold=False)
    elif module == '2.2_candidate_grounded_query_generation':
        sp_running_interface.run_grounded_graph_generation_by_structure(structure_with_2_1_grounded_graph_file, structure_with_2_2_grounded_graph_folder)
        from question_classification import set_operations
        set_operations.run_grounding_graph_update_denotation_graphq(structure_with_2_2_grounded_graph_folder)
        """every grounded graph's f1"""
        kbcqa_evaluation.computed_every_grounded_graph_f1_graphq(structure_with_2_2_grounded_graph_folder)
        kbcqa_evaluation.compute_all_questions_recall(structure_with_2_2_grounded_graph_folder)
    elif module == '2.3_semantic_matching':
        sp_running_interface.run_grounding_graph_score12_match(structure_with_2_2_grounded_graph_folder)
    elif module == '3_evaluation':
        kbcqa_evaluation.run_end_to_end_evaluation(structure_with_2_2_grounded_graph_folder,
                                                   dataset='graphq', output_file='./2020.02.24_output_GraphQ_SP_4_wo_agg_E2E_withnames_all_nonull.json')
    print("end")

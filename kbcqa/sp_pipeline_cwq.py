from common import globals_args
from method_sp import sp_running_interface
from evaluation import kbcqa_evaluation
from common.hand_files import write_structure_file

"""1 parsing"""


def run_ungrounded_graph_from_complexwebquestion(complexquestin_filepath, structure_with_1_ungrounded_cwq_file,node_is_gold=False):
    from datasets_interface.question_interface import complexwebquestion_interface
    complexwebq_list = complexwebquestion_interface.read_complexwebq_question_json(complexquestin_filepath)
    tuples_list = []
    for i, complexwebq_struct in enumerate(complexwebq_list):
        tuples_list.append((complexwebq_struct.ID, complexwebq_struct.question, complexwebq_struct.sparql, complexwebq_struct.answers))
    print(len(tuples_list))
    structure_list = sp_running_interface.run_query_graph_generation(tuples_list=tuples_list, node_is_gold=node_is_gold)
    write_structure_file(structure_list, structure_with_1_ungrounded_cwq_file)


if __name__ == '__main__':
    module = "2.2"
    print('#module:', module)
    complexwebquestion_filepath = globals_args.fn_cwq_file.complexwebquestion_test_dir
    output_path = globals_args.fn_cwq_file.dataset + 'output_cwq_e2e/output_cwq_sp_skeleton_slot_e2e_sp2_v0.1_wo_agg_完成/'
    structure_with_1_ungrounded_graphq_file = output_path + 'structures_with_1_ungrounded_graphs_train_skeleton_0202.json'
    structure_with_2_1_grounded_graph_file = output_path + 'structures_with_2_1_grounded_graph_test_skeleton_0202.json'
    structure_with_2_2_grounded_graph_folder = output_path + '2.2_test_agg/'

    # module
    if module == '1_ungrounded_query_generation':
        run_ungrounded_graph_from_complexwebquestion(complexwebquestion_filepath, structure_with_1_ungrounded_graphq_file, node_is_gold=False)
    elif module == '2.1_entity_linking':
        sp_running_interface.run_grounded_node_grounding_freebase(structure_with_1_ungrounded_graphq_file,
                                                                  structure_with_2_1_grounded_graph_file, linking_is_gold=False)
    elif module == '2.2_candidate_grounded_query_generation':
        sp_running_interface.run_grounded_graph_generation_by_structure(structure_with_2_1_grounded_graph_file,
                                                                        structure_with_2_2_grounded_graph_folder)
        from question_classification import set_operations
        set_operations.run_grounding_graph_update_denotation_cwq(structure_with_2_2_grounded_graph_folder)
        """every grounded graph's f1"""
        kbcqa_evaluation.computed_every_grounded_graph_f1_cwq(structure_with_2_2_grounded_graph_folder)
        kbcqa_evaluation.compute_all_questions_recall(structure_with_2_2_grounded_graph_folder)
    elif module == '2.3_semantic_matching':
        sp_running_interface.run_grounding_graph_score12_match(structure_with_2_2_grounded_graph_folder)
    elif module == '3_evaluation':
        kbcqa_evaluation.run_end_to_end_evaluation(structure_with_2_2_grounded_graph_folder,
                                                   dataset='cwq', output_file='./2021.02.24_output_CWQ_SP_4_wo_agg_E2E_withnames_all_nonull.json')
    print("end")

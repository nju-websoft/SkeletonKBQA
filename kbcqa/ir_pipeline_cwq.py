from common.hand_files import write_structure_file
from common import globals_args
from method_ir import ir_running_interface
from evaluation import kbcqa_evaluation

"""1 parsing"""


def run_topic_entities_from_cwq(filepath, structure_with_1_ungrounded_lcquad_file, node_is_gold=False, linking_is_gold=False):
    from datasets_interface.question_interface import complexwebquestion_interface
    cwq_list = complexwebquestion_interface.read_complexwebq_question_json(filepath)
    tuples_list = []
    for i, cwq_struct in enumerate(cwq_list):
        tuples_list.append((cwq_struct.ID, cwq_struct.question, cwq_struct.sparql, cwq_struct.answers))
    structure_list = ir_running_interface.run_topics_entity_generation_freebase(tuples_list=tuples_list,
                                                                                node_is_gold=node_is_gold,
                                                                                linking_is_gold=linking_is_gold)
    write_structure_file(structure_list, structure_with_1_ungrounded_lcquad_file)


if __name__ == '__main__':
    module = "3_evaluation"
    print('#module:', module)
    filepath = globals_args.fn_cwq_file.complexwebquestion_train_dir
    output_path = globals_args.fn_cwq_file.dataset + 'output_cwq_e2e/output_cwq_ir_skeleton_score12_ir5_v0.1_wo_agg/'
    structure_with_1_ungrounded_cwq_file = output_path + 'structures_with_1_0_train_ir_0202_with_system_nerd.json'
    structure_with_2_0_grounded_lcwq_folder = output_path + '/2.0_test_woagg_1774/'

    if module == '1_node_recognition_and_linking':
        run_topic_entities_from_cwq(filepath, structure_with_1_ungrounded_cwq_file, node_is_gold=False, linking_is_gold=False)
    elif module == '2.1_candidate_grounded_path_generation':
        ir_running_interface.run_candidate_graph_generation(structure_with_1_ungrounded_cwq_file, structure_with_2_0_grounded_lcwq_folder)
        from question_classification import set_operations
        set_operations.run_grounding_graph_update_denotation_cwq(structure_with_2_0_grounded_lcwq_folder)
        """every grounded graph's f1"""
        kbcqa_evaluation.computed_every_grounded_graph_f1_cwq(structure_with_2_0_grounded_lcwq_folder)
        kbcqa_evaluation.compute_all_questions_recall(structure_with_2_0_grounded_lcwq_folder)
    elif module == '2.2_semantic_matching':
        ir_running_interface.run_grounding_graph_score12_match(structure_with_2_0_grounded_lcwq_folder)
    elif module == '3_evaluation':
        kbcqa_evaluation.run_end_to_end_evaluation(structure_with_2_0_grounded_lcwq_folder,
                                                   dataset='cwq', output_file='./2021.03.06_output_cwq_IR_5_v0.1_wo_agg_withnames_all_nonull.json')
    else:
        pass
    print("end")

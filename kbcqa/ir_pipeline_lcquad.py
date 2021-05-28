from common.hand_files import write_structure_file
from common import globals_args
from method_ir import ir_running_interface
from evaluation import kbcqa_evaluation

"""1 parsing"""


def run_topic_entities_from_lcquad(filepath, structure_with_1_ungrounded_lcquad_file, node_is_gold=False, linking_is_gold=False):
    from datasets_interface.question_interface import lcquad_1_0_interface
    lcquad_list = lcquad_1_0_interface.read_train_test_data(filepath)
    tuples_list = []
    for i, lcquad_struct in enumerate(lcquad_list):
        tuples_list.append((lcquad_struct.qid, lcquad_struct.question_normal, lcquad_struct.sparql, None))
    structure_list = ir_running_interface.run_topics_entity_generation_dbpedia(tuples_list=tuples_list,
                                                                               node_is_gold=node_is_gold,
                                                                               linking_is_gold=linking_is_gold)
    write_structure_file(structure_list, structure_with_1_ungrounded_lcquad_file)


if __name__ == '__main__':
    module = "3_evaluation"
    print('#module:', module)
    filepath = globals_args.fn_lcquad_file.lcquad_train_dir
    output_path = globals_args.fn_lcquad_file.dataset + 'output_lcquad_e2e/output_lcquad_ir_skeleton_score12_ir5/'
    structure_with_1_ungrounded_lcquad_file = output_path + 'structures_with_1_0_train_ir_0201.json'
    structure_with_2_0_grounded_lcquad_folder = output_path + '/2.0_test_669/'

    if module == '1_node_recognition_and_linking':
        run_topic_entities_from_lcquad(filepath, structure_with_1_ungrounded_lcquad_file, node_is_gold=False, linking_is_gold=False)
    elif module == '2.1_candidate_grounded_path_generation':
        ir_running_interface.run_candidate_graph_generation(structure_with_1_ungrounded_lcquad_file, structure_with_2_0_grounded_lcquad_folder)
        """every grounded graph's f1"""
        kbcqa_evaluation.computed_every_grounded_graph_f1_lcquad(structure_with_2_0_grounded_lcquad_folder)
        kbcqa_evaluation.compute_all_questions_recall(structure_with_2_0_grounded_lcquad_folder)
    elif module == '2.2_semantic_matching':
        ir_running_interface.run_grounding_graph_score12_match(structure_with_2_0_grounded_lcquad_folder)
    elif module == '3_evaluation':
        kbcqa_evaluation.run_end_to_end_evaluation(structure_with_2_0_grounded_lcquad_folder,
                                                   dataset='lcquad', output_file='./2021.02.05_output_LC_IR_5_E2E_withnames_all_nonull.json')
    else:
        pass
    print("end")

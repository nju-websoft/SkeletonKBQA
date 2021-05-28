from common.hand_files import write_structure_file
from common import globals_args
from method_sp import sp_running_interface
from evaluation import kbcqa_evaluation

"""1 parsing"""


def run_ungrounded_graph_from_lcquad(filepath, structure_with_1_ungrounded_lcquad_file, node_is_gold=False):
    from datasets_interface.question_interface import lcquad_1_0_interface
    lcquad_list = lcquad_1_0_interface.read_train_test_data(filepath)
    tuples_list = []
    for i, lcquad_struct in enumerate(lcquad_list):
        tuples_list.append((lcquad_struct.qid, lcquad_struct.question_normal, lcquad_struct.sparql, None))
    print(len(tuples_list))
    structure_list = sp_running_interface.run_query_graph_generation(tuples_list=tuples_list, node_is_gold=node_is_gold)
    write_structure_file(structure_list, structure_with_1_ungrounded_lcquad_file)


if __name__ == '__main__':
    module = "3_evaluation"
    print('#module:', module)
    filepath = globals_args.fn_lcquad_file.lcquad_train_dir
    output_path = globals_args.fn_lcquad_file.dataset + 'output_lcquad_e2e/output_lcquad_sp_skeleton_sp_multi_strategy_e2e_sp1_完成'
    structure_with_1_ungrounded_lcquad_file = output_path + '/structures_with_1_grounded_graphs_dep_train_0201.json'
    structure_with_2_1_grounded_lcquad_file = output_path + '/structures_with_2_1_grounded_graphs_dep_train_0201.json'
    structure_with_2_2_grounded_lcquad_folder = output_path + '/2.2_test_597/'

    if module == '1_ungrounded_query_generation':
        run_ungrounded_graph_from_lcquad(filepath, structure_with_1_ungrounded_lcquad_file, node_is_gold=False)
    elif module == '2.1_entity_linking':
        sp_running_interface.run_grounded_node_grounding_dbpedia(structure_with_1_ungrounded_lcquad_file,
                                                                 structure_with_2_1_grounded_lcquad_file, linking_is_gold=False)
    elif module == '2.2_candidate_grounded_query_generation':
        sp_running_interface.run_grounded_graph_generation_by_structure(structure_with_2_1_grounded_lcquad_file,
                                                                        structure_with_2_2_grounded_lcquad_folder)
        """every grounded graph's f1"""
        kbcqa_evaluation.computed_every_grounded_graph_f1_lcquad(structure_with_2_2_grounded_lcquad_folder)
        kbcqa_evaluation.compute_all_questions_recall(structure_with_2_2_grounded_lcquad_folder)
    elif module == '2.3_semantic_matching':
        sp_running_interface.run_grounding_graph_score12_match(structure_with_2_2_grounded_lcquad_folder)
    elif module == '3_evaluation':
        kbcqa_evaluation.run_end_to_end_evaluation(structure_with_2_2_grounded_lcquad_folder,
                                                   dataset='lcquad', output_file='./2021.02.22_output_LC_SP_1_E2E_withnames_all_nonull.json')
    else:
        pass
    print("end")

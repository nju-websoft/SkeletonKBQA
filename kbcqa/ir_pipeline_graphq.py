from common.hand_files import write_structure_file
from common import globals_args
from method_ir import ir_running_interface
from evaluation import kbcqa_evaluation

"""1 parsing"""


def run_topic_entities_from_graphq(graph_questions_filepath, structure_with_1_ungrounded_graphq_file, node_is_gold=False, linking_is_gold=False):
    from datasets_interface.question_interface import graphquestion_interface
    graph_questions_struct = graphquestion_interface.read_graph_question_json(graph_questions_filepath)
    tuples_list = []
    for i, graphquestion in enumerate(graph_questions_struct):
        tuples_list.append((graphquestion.qid, graphquestion.question, graphquestion.graph_query, graphquestion.answer_mid))
    print(len(tuples_list))
    structure_list = ir_running_interface.run_topics_entity_generation_freebase(tuples_list=tuples_list,
                                                                                node_is_gold=node_is_gold,
                                                                                linking_is_gold=linking_is_gold)
    write_structure_file(structure_list, structure_with_1_ungrounded_graphq_file)


if __name__ == '__main__':
    module = "3_evaluation"
    print('#module:', module)
    graph_questions_filepath = globals_args.fn_graph_file.graphquestions_testing_dir
    output_path = globals_args.fn_graph_file.dataset + 'output_graphq_e2e/output_graphq_ir_dep_score12_ir6_v0.1_wo_agg/'
    structure_with_1_ungrounded_graphq_file = output_path + 'structures_with_1_0_train_ir_0130_with_system_nerd.json'
    structure_with_2_0_grounded_graphq_folder = output_path + '/2.0_test_1149/'

    if module == '1_node_recognition_and_linking':
        run_topic_entities_from_graphq(graph_questions_filepath, structure_with_1_ungrounded_graphq_file, node_is_gold=False, linking_is_gold=False)
    elif module == '2.1_candidate_grounded_path_generation':
        ir_running_interface.run_candidate_graph_generation(structure_with_1_ungrounded_graphq_file, structure_with_2_0_grounded_graphq_folder)
        from question_classification import set_operations
        set_operations.run_grounding_graph_update_denotation_graphq(structure_with_2_0_grounded_graphq_folder)
        """every grounded graph's f1"""
        kbcqa_evaluation.computed_every_grounded_graph_f1_graphq(structure_with_2_0_grounded_graphq_folder)
        kbcqa_evaluation.compute_all_questions_recall(structure_with_2_0_grounded_graphq_folder)
    elif module == '2.2_semantic_matching':
        ir_running_interface.run_grounding_graph_score12_match(structure_with_2_0_grounded_graphq_folder)
    elif module == '3_evaluation':
        kbcqa_evaluation.run_end_to_end_evaluation(structure_with_2_0_grounded_graphq_folder,
                                                   dataset='graphq', output_file='./2021.03.03_output_GraphQ_IR_6_v0.1_wo_agg_E2E_withnames_all_nonull.json')
    else:
        pass
    print("end")

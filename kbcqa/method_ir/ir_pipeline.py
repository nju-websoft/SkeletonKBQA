from common.hand_files import write_structure_file
from common import globals_args
from method_ir import ir_module
from evaluation import kbcqa_evaluation
from argparse import ArgumentParser
from question_classification import set_operations


"""lcquad parsing"""


def run_topic_entities_from_lcquad(filepath, structure_with_1_ungrounded_lcquad_file, node_is_gold=False, linking_is_gold=False):
    from datasets_interface.question_interface import lcquad_1_0_interface
    lcquad_list = lcquad_1_0_interface.read_train_test_data(filepath)
    tuples_list = []
    for i, lcquad_struct in enumerate(lcquad_list):
        tuples_list.append((lcquad_struct.qid, lcquad_struct.question_normal, lcquad_struct.sparql, None))
    structure_list = ir_module.run_topics_entity_generation_dbpedia(tuples_list=tuples_list,
                                                                    node_is_gold=node_is_gold,
                                                                    linking_is_gold=linking_is_gold,
                                                                    q_mode='lcquad')
    write_structure_file(structure_list, structure_with_1_ungrounded_lcquad_file)


"""graphq parsing"""


def run_topic_entities_from_graphq(graph_questions_filepath, structure_with_1_ungrounded_graphq_file, node_is_gold=False, linking_is_gold=False):
    from datasets_interface.question_interface import graphquestion_interface
    graph_questions_struct = graphquestion_interface.read_graph_question_json(graph_questions_filepath)
    tuples_list = []
    for i, graphquestion in enumerate(graph_questions_struct):
        tuples_list.append((graphquestion.qid, graphquestion.question, graphquestion.graph_query, graphquestion.answer_mid))
    print(len(tuples_list))
    structure_list = ir_module.run_topics_entity_generation_freebase(tuples_list=tuples_list, node_is_gold=node_is_gold,
                                                                     linking_is_gold=linking_is_gold,
                                                                     q_mode='graphq')
    write_structure_file(structure_list, structure_with_1_ungrounded_graphq_file)


"""cwq parsing"""


def run_topic_entities_from_cwq(filepath, structure_with_1_ungrounded_cwq_file, node_is_gold=False, linking_is_gold=False):
    from datasets_interface.question_interface import complexwebquestion_interface
    cwq_list = complexwebquestion_interface.read_complexwebq_question_json(filepath)
    tuples_list = []
    for i, cwq_struct in enumerate(cwq_list):
        tuples_list.append((cwq_struct.ID, cwq_struct.question, cwq_struct.sparql, cwq_struct.answers))
    structure_list = ir_module.run_topics_entity_generation_freebase(tuples_list=tuples_list,
                                                                     node_is_gold=node_is_gold,
                                                                     linking_is_gold=linking_is_gold,
                                                                     q_mode='cwq')
    write_structure_file(structure_list, structure_with_1_ungrounded_cwq_file)


if __name__ == '__main__':
    parser = ArgumentParser(description="arguments")
    parser.add_argument('--module', type=str, help='1_node_recognition_and_linking, 2.1_candidate_grounded_path_generation, '
                                                   '2.2_semantic_matching, 3_evaluation', default='1_node_recognition_and_linking')
    parser.add_argument('--node_is_gold', type=bool, help='whether node is gold', default=False)
    parser.add_argument('--linking_is_gold', type=bool, help='whether linking is gold', default=False)
    parser.add_argument('--q_mode', type=str, help='lcquad, cwq, graphq', default='lcquad')
    parser.add_argument('--dataset', type=str, help='train, test', default='test')
    parser.add_argument('--output_folder', type=str, help='output folder', default='output_lcquad_e2e')
    parser.add_argument('--output_configuration', type=str, help='output configuration', default='output_lcquad_ir_skeleton_score12_ir5')
    parser.add_argument('--ungrounded_file', type=str, help='ungrounded file', default='structures_with_1_0_train_ir.json')
    parser.add_argument('--grounded_folder', type=str, help='grounded folder', default='2.0_test')
    parser.add_argument('--output_result_file', type=str, help='the output file', default='2021.02.05_output_LC_IR_5_E2E.json')
    args = parser.parse_args()

    print('#module:', args.module)

    assert args.dataset in ['test', 'dev', 'train']
    assert args.q_mode in ['lcquad', 'graphq', 'cwq']
    filepath, ungrounded_file, grounded_folder, output_result_file = None, None, None, None
    if args.q_mode == 'lcquad':
        if args.dataset == 'test':
            filepath = globals_args.fn_lcquad_file.lcquad_test_dir
        elif args.dataset == 'train':
            filepath = globals_args.fn_lcquad_file.lcquad_train_dir
        ungrounded_file = globals_args.fn_lcquad_file.dataset + args.output_folder + '/' + args.output_configuration + '/' + args.ungrounded_file
        grounded_folder = globals_args.fn_lcquad_file.dataset + args.output_folder + '/' + args.output_configuration + '/' + args.grounded_folder + '/'
        output_result_file = globals_args.fn_lcquad_file.dataset + args.output_folder + '/' + args.output_configuration + '/' + args.output_result_file

    elif args.q_mode == 'graphq':
        if args.dataset == 'test':
            filepath = globals_args.fn_graph_file.graphquestions_testing_dir
        elif args.dataset == 'train':
            filepath = globals_args.fn_graph_file.graphquestions_training_dir
        ungrounded_file = globals_args.fn_graph_file.dataset + args.output_folder + '/' + args.ungrounded_file
        grounded_folder = globals_args.fn_graph_file.dataset + args.output_folder + '/' + args.grounded_folder
        output_result_file = globals_args.fn_graph_file.dataset + args.output_folder + '/' + args.output_configuration + '/' + args.output_result_file

    elif args.q_mode == 'cwq':
        if args.dataset == 'test':
            filepath = globals_args.fn_cwq_file.complexwebquestion_test_dir
        elif args.dataset == 'dev':
            filepath = globals_args.fn_cwq_file.complexwebquestion_dev_dir
        elif args.dataset == 'train':
            filepath = globals_args.fn_cwq_file.complexwebquestion_train_dir
        ungrounded_file = globals_args.fn_cwq_file.dataset + args.output_folder + '/' + args.ungrounded_file
        grounded_folder = globals_args.fn_cwq_file.dataset + args.output_folder + '/' + args.grounded_folder
        output_result_file = globals_args.fn_cwq_file.dataset + args.output_folder + '/' + args.output_configuration + '/' + args.output_result_file


    """running SIR pipeline"""
    if args.module == '1_node_recognition_and_linking':
        if args.q_mode == 'lcquad':
            run_topic_entities_from_lcquad(filepath, ungrounded_file, node_is_gold=args.node_is_gold, linking_is_gold=args.linking_is_gold)
        elif args.q_mode == 'graphq':
            run_topic_entities_from_graphq(filepath, ungrounded_file, node_is_gold=args.node_is_gold, linking_is_gold=args.linking_is_gold)
        elif args.q_mode == 'cwq':
            run_topic_entities_from_cwq(filepath, ungrounded_file, node_is_gold=args.node_is_gold, linking_is_gold=args.linking_is_gold)

    elif args.module == '2.1_candidate_grounded_path_generation':
        ir_module.run_candidate_graph_generation(ungrounded_file, grounded_folder, q_mode=args.q_mode)
        """every grounded graph's f1"""
        if args.q_mode == 'lcquad':
            kbcqa_evaluation.computed_every_grounded_graph_f1_lcquad(grounded_folder)
        elif args.q_mode == 'graphq':
            set_operations.run_grounding_graph_update_denotation_graphq(grounded_folder)
            kbcqa_evaluation.computed_every_grounded_graph_f1_graphq(grounded_folder)
        elif args.q_mode == 'cwq':
            set_operations.run_grounding_graph_update_denotation_cwq(grounded_folder)
            kbcqa_evaluation.computed_every_grounded_graph_f1_cwq(grounded_folder)
        # kbcqa_evaluation.compute_all_questions_recall(grounded_folder)

    elif args.module == '2.2_semantic_matching':
        ir_module.run_grounding_graph_score12_match(grounded_folder, q_mode=args.q_mode)

    elif args.module == '3_evaluation':
        kbcqa_evaluation.run_end_to_end_evaluation(grounded_folder, q_mode=args.q_mode, output_file=output_result_file)

    else:
        pass
    print("end")

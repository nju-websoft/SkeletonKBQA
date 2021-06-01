from common.hand_files import write_structure_file
from common import globals_args
from method_sp import sp_modules
from evaluation import kbcqa_evaluation
from argparse import ArgumentParser
from question_classification import set_operations


"""lcquad parsing"""


def run_ungrounded_graph_from_lcquad(filepath, structure_with_1_ungrounded_lcquad_file, node_is_gold=False, parser_mode='skeleton'):
    from datasets_interface.question_interface import lcquad_1_0_interface
    lcquad_list = lcquad_1_0_interface.read_train_test_data(filepath)
    tuples_list = []
    for i, lcquad_struct in enumerate(lcquad_list):
        tuples_list.append((lcquad_struct.qid, lcquad_struct.question_normal, lcquad_struct.sparql, None))
    print(len(tuples_list))
    structure_list = sp_modules.run_query_graph_generation(tuples_list=tuples_list, node_is_gold=node_is_gold, parser_mode=parser_mode, q_mode='lcquad')
    write_structure_file(structure_list, structure_with_1_ungrounded_lcquad_file)


"""graphq parsing"""


def run_ungrounded_graph_from_graphq(graph_questions_filepath, output_file, node_is_gold=False, parser_mode='skeleton'):
    from datasets_interface.question_interface import graphquestion_interface
    graph_questions_struct = graphquestion_interface.read_graph_question_json(graph_questions_filepath)
    tuples_list = []
    for i, graphquestion in enumerate(graph_questions_struct):
        tuples_list.append((graphquestion.qid, graphquestion.question, graphquestion.graph_query, graphquestion.answer))
    print(len(tuples_list))
    structure_list = sp_modules.run_query_graph_generation(tuples_list=tuples_list, node_is_gold=node_is_gold, parser_mode=parser_mode, q_mode='graphq')
    write_structure_file(structure_list, output_file)


"""cwq parsing"""


def run_ungrounded_graph_from_complexwebquestion(complexquestin_filepath, structure_with_1_ungrounded_cwq_file,node_is_gold=False, parser_mode='skeleton'):
    from datasets_interface.question_interface import complexwebquestion_interface
    complexwebq_list = complexwebquestion_interface.read_complexwebq_question_json(complexquestin_filepath)
    tuples_list = []
    for i, complexwebq_struct in enumerate(complexwebq_list):
        tuples_list.append((complexwebq_struct.ID, complexwebq_struct.question, complexwebq_struct.sparql, complexwebq_struct.answers))
    print(len(tuples_list))
    structure_list = sp_modules.run_query_graph_generation(tuples_list=tuples_list, node_is_gold=node_is_gold, parser_mode=parser_mode, q_mode='cwq')
    write_structure_file(structure_list, structure_with_1_ungrounded_cwq_file)


if __name__ == '__main__':
    parser = ArgumentParser(description="arguments")
    parser.add_argument('--module', type=str, help='1_ungrounded_query_generation, 2.1_entity_linking, 2.2_candidate_grounded_query_generation, '
                                                   '2.3_semantic_matching, 3_evaluation', default='1_ungrounded_query_generation')
    parser.add_argument('--node_is_gold', type=bool, help='whether node is gold', default=False)
    parser.add_argument('--linking_is_gold', type=bool, help='whether linking is gold', default=False)
    parser.add_argument('--q_mode', type=str, help='lcquad, cwq, graphq', default='lcquad')
    parser.add_argument('--dataset', type=str, help='train, test', default='test')
    parser.add_argument('--parser_mode', type=str, help='skeleton or dependency', default='skeleton')
    parser.add_argument('--output_folder', type=str, help='output folder', default='output_lcquad_e2e')
    parser.add_argument('--output_configuration', type=str, help='output configuration', default='output_lcquad_sp_skeleton_slot_e2e_sp4')
    parser.add_argument('--ungrounded_file', type=str, help='ungrounded file', default='structures_with_1_grounded_graph_skeleton_test.json')
    parser.add_argument('--_2_1_grounded_file', type=str, help='lcquad, cwq, graphq', default='structures_with_2_1_grounded_graph_skeleton_test.json')
    parser.add_argument('--grounded_folder', type=str, help='grounded folder', default='2.0_test')
    parser.add_argument('--output_result_file', type=str, help='the output file', default='2021.02.22_output_LC_SP_1_E2E.json')
    args = parser.parse_args()

    print('#module:', args.module)

    assert args.dataset in ['test', 'dev', 'train']
    assert args.q_mode in ['lcquad', 'graphq', 'cwq']
    filepath, ungrounded_file, _2_1_grounded_file, grounded_folder, output_result_file = None, None, None, None, None
    if args.q_mode == 'lcquad':
        if args.dataset == 'test':
            filepath = globals_args.fn_lcquad_file.lcquad_test_dir
        elif args.dataset == 'train':
            filepath = globals_args.fn_lcquad_file.lcquad_train_dir
        ungrounded_file = globals_args.fn_lcquad_file.dataset + args.output_folder + '/' + args.output_configuration + '/' + args.ungrounded_file
        _2_1_grounded_file = globals_args.fn_lcquad_file.dataset + args.output_folder + '/' + args._2_1_grounded_file
        grounded_folder = globals_args.fn_lcquad_file.dataset + args.output_folder + '/' + args.output_configuration + '/' + args.grounded_folder + '/'
        output_result_file = globals_args.fn_lcquad_file.dataset + args.output_folder + '/' + args.output_configuration + '/' + args.output_result_file

    elif args.q_mode == 'graphq':
        if args.dataset == 'test':
            filepath = globals_args.fn_graph_file.graphquestions_testing_dir
        elif args.dataset == 'train':
            filepath = globals_args.fn_graph_file.graphquestions_training_dir
        ungrounded_file = globals_args.fn_graph_file.dataset + args.output_folder + '/' + args.ungrounded_file
        _2_1_grounded_file = globals_args.fn_graph_file.dataset + args.output_folder + '/' + args._2_1_grounded_file
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
        _2_1_grounded_file = globals_args.fn_cwq_file.dataset + args.output_folder + '/' + args._2_1_grounded_file
        grounded_folder = globals_args.fn_cwq_file.dataset + args.output_folder + '/' + args.grounded_folder
        output_result_file = globals_args.fn_cwq_file.dataset + args.output_folder + '/' + args.output_configuration + '/' + args.output_result_file


    """running SSP pipeline"""
    if args.module == '1_ungrounded_query_generation':
        if args.q_mode == 'lcquad':
            run_ungrounded_graph_from_lcquad(filepath, ungrounded_file, node_is_gold=args.node_is_gold, parser_mode=args.parser_mode)
        elif args.q_mode == 'graphq':
            run_ungrounded_graph_from_graphq(filepath, ungrounded_file, node_is_gold=args.node_is_gold, parser_mode=args.parser_mode)
        elif args.q_mode == 'cwq':
            run_ungrounded_graph_from_complexwebquestion(filepath, ungrounded_file, node_is_gold=args.node_is_gold, parser_mode=args.parser_mode)

    elif args.module == '2.1_entity_linking':
        if args.q_mode == 'lcquad':
            sp_modules.run_grounded_node_grounding_dbpedia(ungrounded_file, _2_1_grounded_file, linking_is_gold=args.linking_is_gold)
        elif args.q_mode in ['graphq','cwq']:
            sp_modules.run_grounded_node_grounding_freebase(ungrounded_file, _2_1_grounded_file, linking_is_gold=args.linking_is_gold, q_mode=args.q_mode)

    elif args.module == '2.2_candidate_grounded_query_generation':
        sp_modules.run_grounded_graph_generation_by_structure(_2_1_grounded_file, grounded_folder, q_mode=args.q_mode)
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

    elif args.module == '2.3_semantic_matching':
        sp_modules.run_grounding_graph_score12_match(grounded_folder, q_mode=args.q_mode)

    elif args.module == '3_evaluation':
        kbcqa_evaluation.run_end_to_end_evaluation(grounded_folder, q_mode=args.q_mode, output_file=args.output_result_file)
    else:
        pass
    print("end")


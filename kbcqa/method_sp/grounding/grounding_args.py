from common import globals_args
from method_sp.grounding import grounding_utils
import os
from common import hand_files


q_mode = globals_args.argument_parser.q_mode

# 2.2 args
if q_mode == 'cwq':
    oracle_file_root = globals_args.fn_cwq_file.grounded_graph_file+'result/'
    oracle_all_files_path_names = os.listdir(oracle_file_root)
    literal_to_id_map = grounding_utils.read_literal_to_id_map(file_root=globals_args.fn_cwq_file.grounded_graph_file)
    kb_relations = hand_files.read_set(globals_args.kb_freebase_latest_file.freebase_relations_file)

    mediators_instances_set = hand_files.read_set(globals_args.kb_freebase_latest_file.mediators_instances_file)
    schema_lines_list = hand_files.read_list(globals_args.kb_freebase_latest_file.schema_file)
    property_reverse_dict = hand_files.read_dict(globals_args.kb_freebase_latest_file.freebase_reverse_property)
    literal_property_dict = hand_files.read_dict(globals_args.kb_freebase_latest_file.freebase_literal_property)

elif q_mode == 'graphq':
    oracle_file_root = globals_args.fn_graph_file.grounded_graph_file+'result/'
    oracle_all_files_path_names = os.listdir(oracle_file_root)
    literal_to_id_map = grounding_utils.read_literal_to_id_map(file_root=globals_args.fn_graph_file.grounded_graph_file)
    kb_relations = hand_files.read_set(globals_args.kb_freebase_en_2013.freebase_relations_file)

    mediators_instances_set = hand_files.read_set(globals_args.kb_freebase_en_2013.mediators_instances_file)
    schema_lines_list = hand_files.read_list(globals_args.kb_freebase_en_2013.schema_file)
    property_reverse_dict = hand_files.read_dict(globals_args.kb_freebase_en_2013.freebase_reverse_property_file)
    literal_property_dict = hand_files.read_dict(globals_args.kb_freebase_en_2013.freebase_literal_property)

elif q_mode == 'lcquad':
    oracle_file_root = globals_args.fn_lcquad_file.grounded_graph_file+'result/'
    oracle_all_files_path_names = os.listdir(oracle_file_root)
    kb_relations = hand_files.read_list_yuanshi(globals_args.kb_dbpedia_201604_file.dbpedia_relations_file)


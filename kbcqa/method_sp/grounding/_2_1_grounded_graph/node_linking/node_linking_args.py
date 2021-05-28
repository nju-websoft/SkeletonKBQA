from common import globals_args
from common import hand_files

wh_words_set = {"what", "which", "whom", "who", "when", "where", "why", "how", "how many", "how large", "how big"}
kb_mode = globals_args.kb_mode

if kb_mode == 'kb_dbpedia_201604':
    label_to_iris_dict_dict = hand_files.read_pickle(globals_args.kb_dbpedia_201604_file.label_lexicon_path)
    wikipage_to_iris_dict_dict = hand_files.read_pickle(globals_args.kb_dbpedia_201604_file.pageRedirects_lexicon_path)
    wikilinkText_to_iris_dict_dict = hand_files.read_pickle(globals_args.kb_dbpedia_201604_file.wikiLinkText_lexicon_path)

elif kb_mode == 'kb_freebase_latest':
    entity_list_file = globals_args.kb_freebase_latest_file.entity_list_file
    surface_map_file = globals_args.kb_freebase_latest_file.surface_map_file
    entity_index_prefix = globals_args.kb_freebase_latest_file.entity_index_prefix

    freebase_relations = globals_args.kb_freebase_latest_file.freebase_relations_file
    quotation_dict = hand_files.read_dict(globals_args.kb_freebase_latest_file.quotation_file)
    mediatortypes = hand_files.read_set(globals_args.kb_freebase_latest_file.mediatortypes_file)

elif kb_mode == 'kb_freebase_en_2013':
    freebase_graph_name_entity_file = globals_args.kb_freebase_en_2013.freebase_graph_name_entity
    freebase_graph_alias_entity_file = globals_args.kb_freebase_en_2013.freebase_graph_alias_entity
    clueweb_mention_pro_entity_file = globals_args.kb_freebase_en_2013.clueweb_mention_pro_entity
    graphquestions_train_friendlyname_entity_file = globals_args.kb_freebase_en_2013.graphquestions_train_friendlyname_entity

    freebase_relations = hand_files.read_set(globals_args.kb_freebase_en_2013.freebase_relations_file)
    quotation_dict = dict()
    mediatortypes = hand_files.read_set(globals_args.kb_freebase_en_2013.mediatortypes_file)

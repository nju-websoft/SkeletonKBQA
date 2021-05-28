
class KB_DBpedia_201604():

    def __init__(self, root):
        self.dataset = root + '/kb_dbpedia_201604/'
        """lexicon"""
        self.label_lexicon_path = self.dataset+'2019.07.16_dbpedia201604_label_lexicon.pt'
        self.wikiLinkText_lexicon_path = self.dataset+'2019.07.20_dbpedia201604_wikiPageWikiLinkText_lexicon.pt'
        self.pageRedirects_lexicon_path = self.dataset+'2019.07.20_dbpedia201604_wikiPageRedirects_lexicon.pt'
        """relation information"""
        self.property_level_words_dbo_datatype = self.dataset + "relortype_level_words_dbo_datatype.json"
        self.property_level_words_dbo_object = self.dataset + "relortype_level_words_dbo_object.json"
        self.property_level_words_dbp = self.dataset + "relortype_level_words_dbp.json"
        self.dbpedia_relations_file = self.dataset+"dbpedia_relations"


class KB_Freebase_Latest():

    def __init__(self, root):
        self.dataset = root + '/kb_freebase_latest/'
        """entity linking"""
        self.entity_list_file = self.dataset + "el_aqqu_mid_vocab/entity-list"
        self.surface_map_file = self.dataset + "el_aqqu_mid_vocab/entity-surface-map"
        self.entity_index_prefix = self.dataset + "el_aqqu_mid_vocab/entity-index"
        """mediator information"""
        self.mediatortypes_file = self.dataset + "mediators.tsv"
        self.mediators_instances_file = self.dataset + "freebase_all_mediators_instances"
        """quotation"""
        self.quotation_file = self.dataset + 'freebase_all_quotations_name'
        """relation information"""
        self.freebase_relations_file = self.dataset+"freebase_relations"
        self.freebase_reverse_property = self.dataset + "freebase_reverse_property"
        self.freebase_literal_property = self.dataset + "freebase_schema_literal"
        """schema"""
        self.schema_file = self.dataset + "freebase_schema"


class KB_Freebase_en_2013():

    def __init__(self, root):
        self.dataset = root + '/kb_freebase_en_2013/'
        """entity linking"""
        self.freebase_graph_name_entity = self.dataset + 'el_en_vocab/graphq201306_nameentity_handled'
        self.freebase_graph_alias_entity = self.dataset + 'el_en_vocab/graphq201306_aliasentity_handled'
        self.graphquestions_train_friendlyname_entity = self.dataset + 'el_en_vocab/graphquestions_train_friendlyname_entity_handled'
        self.clueweb_mention_pro_entity = self.dataset + 'el_en_vocab/clueweb_name_entity_pro_handled'
        """mediator information"""
        self.mediatortypes_file = self.dataset + "mediatortypes"
        self.mediators_instances_file = self.dataset + "freebase_all_mediators_instances"
        """relation information"""
        self.freebase_relations_file = self.dataset+"freebase_relations"
        self.freebase_reverse_property_file = self.dataset + "freebase_reverse_property"
        self.freebase_literal_property = self.dataset + "freebase_schema_literal"
        """schema"""
        self.schema_file = self.dataset + "freebase_schema"




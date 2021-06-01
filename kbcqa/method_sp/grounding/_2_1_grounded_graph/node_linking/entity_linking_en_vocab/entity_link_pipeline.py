# -*- coding: utf-8 -*-
# from common.globals_args import fn_graph_file, fn_cwq_file
from method_sp.grounding._2_1_grounded_graph.node_linking.entity_linking_en_vocab.entity_linker import EntityLinker
from common.hand_files import read_dict, read_dict_dict


class EntityVocabulary():
    '''four lexicons  entity_vocabulary'''

    def __init__(self, freebase_graph_name_entity_file, freebase_graph_alias_entity_file,
                 graphquestions_train_friendlyname_entity_file, clueweb_mention_pro_entity_file):
        self.freebase_graph_name_entity = read_dict(freebase_graph_name_entity_file)
        self.freebase_graph_alias_entity = read_dict(freebase_graph_alias_entity_file)
        if graphquestions_train_friendlyname_entity_file != '':
            self.graphquestions_train_friendlyname_entity = read_dict(graphquestions_train_friendlyname_entity_file)
        else:
            self.graphquestions_train_friendlyname_entity = dict()
        self.clueweb_mention_pro_entity = read_dict_dict(clueweb_mention_pro_entity_file)


class EntityLinkPipeline():
    '''entity linking'''

    def __init__(self, freebase_graph_name_entity_file, freebase_graph_alias_entity_file,
                 graphquestions_train_friendlyname_entity_file, clueweb_mention_pro_entity_file):
        self.entity_vocabulary = EntityVocabulary(freebase_graph_name_entity_file, freebase_graph_alias_entity_file,
                                                  graphquestions_train_friendlyname_entity_file, clueweb_mention_pro_entity_file)

    def get_indexrange_entity_el_pro_one_mention(self, phrase, top_k=10):
        '''
        :param indexrange_phrase: 3\t7 how many tennis tournament championships
        :return: 9	9 {'en.america': 1.4285881881259241,}
        '''
        el = EntityLinker(self.entity_vocabulary)
        return el.get_indexrange_entity_pros_by_mention(phrase, top_k)


if __name__ == '__main__':
    from method_sp.grounding._2_1_grounded_graph.node_linking import node_linking_args
    elp = EntityLinkPipeline(freebase_graph_name_entity_file=node_linking_args.freebase_graph_name_entity_file,
                         freebase_graph_alias_entity_file=node_linking_args.freebase_graph_alias_entity_file,
                         graphquestions_train_friendlyname_entity_file=node_linking_args.graphquestions_train_friendlyname_entity_file,
                         clueweb_mention_pro_entity_file=node_linking_args.clueweb_mention_pro_entity_file)

    print (elp.get_indexrange_entity_el_pro_one_mention('america', top_k=100))

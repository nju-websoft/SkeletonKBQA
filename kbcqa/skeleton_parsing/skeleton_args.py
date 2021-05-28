from common import globals_args
from skeleton_parsing.nltk_tools import NLTK_NLP


class BertArgs():

    def __init__(self, root, mode):
        # uncased model
        self.bert_base_uncased_model = root + '/pre_train_models/bert-base-uncased.tar.gz'
        self.bert_base_uncased_tokenization = root + '/pre_train_models/bert-base-uncased-vocab.txt'
        # cased model
        self.bert_base_cased_model = root + '/pre_train_models/bert-base-cased.tar.gz'
        self.bert_base_cased_tokenization = root + '/pre_train_models/bert-base-cased-vocab.txt'
        if mode == 'cwq':
            self.get_cwq_args(root=root)
        elif mode == 'graphq':
            self.get_graphq_args(root=root)
        elif mode == 'lcquad':
            self.get_lcquad_args(root=root)
        else:
            pass

    def get_cwq_args(self, root):
        root = root + '/dataset_cwq_1_1/fine_tuning_models_cwq_0107_v0.2/'
        self.fine_tuning_headword_squad_F_model = root + 'debug_headwords_1222_squad_F/pytorch_model.bin'
        self.fine_tuning_relation_classifier_E_model = root + 'debug_relation_classifier_1222_E/pytorch_model.bin'
        self.fine_tuning_sequence_classifier_B_model = root + 'debug_simplification_1222_B/pytorch_model.bin'
        self.fine_tuning_redundancy_span_D_model = root + 'debug_redundancy_1222_D/pytorch_model.bin'

        self.fine_tuning_token_classifier_C_model = root + 'debug_node_3_1222_C/pytorch_model.bin'
        self.fine_tuning_qtype_classifier_G_model = root + 'debug_questiontype_1222_G/pytorch_model.bin'
        self.fine_tuning_qtype_superlative_classifier_G_model = root + 'debug_superlativetype_1222_I/pytorch_model.bin'
        self.fine_tuning_qtype_comparative_classifier_G_model = root + 'debug_comparativetype_1222_H/pytorch_model.bin'

    def get_lcquad_args(self, root):
        root = root + '/dataset_lcquad_1_0/fine_tuning_models_lcquad_1217_v0.2/'
        self.fine_tuning_headword_squad_F_model = root + 'debug_headwords_1222_squad_F/pytorch_model.bin'
        self.fine_tuning_relation_classifier_E_model = root + 'debug_relation_classifier_1222_E/pytorch_model.bin'
        self.fine_tuning_sequence_classifier_B_model = root + 'debug_simplification_1222_B/pytorch_model.bin'
        self.fine_tuning_redundancy_span_D_model = root + 'debug_redundancy_1222_D/pytorch_model.bin'

        self.fine_tuning_token_classifier_C_model = root + 'debug_node_3_1222_C/pytorch_model.bin'
        self.fine_tuning_qtype_classifier_G_model = root + 'debug_questiontype_1222_G/pytorch_model.bin'

    def get_graphq_args(self, root):
        root = root + '/dataset_graphquestions/fine_tuning_models_graphq_1228_v0.3/'
        self.fine_tuning_headword_squad_F_model = root + 'debug_headwords_1222_squad_F/pytorch_model.bin'
        self.fine_tuning_relation_classifier_E_model = root + 'debug_relation_classifier_1222_E/pytorch_model.bin'
        self.fine_tuning_sequence_classifier_B_model = root + 'debug_simplification_1222_B/pytorch_model.bin'
        self.fine_tuning_redundancy_span_D_model = root + 'debug_redundancy_1222_D/pytorch_model.bin'

        self.fine_tuning_token_classifier_C_model = root + 'debug_node_3_1222_C/pytorch_model.bin'
        self.fine_tuning_qtype_classifier_G_model = root + 'debug_questiontype_1222_G/pytorch_model.bin'


bert_args = BertArgs(globals_args.root, globals_args.q_mode)
nltk_nlp = NLTK_NLP(globals_args.argument_parser.ip_port)


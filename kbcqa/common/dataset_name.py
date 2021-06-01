from common.hand_files import read_ngram_el_grounding_result


class LCQuADFileName():

    def __init__(self, root):
        self.dataset = root + '/dataset_lcquad_1_0/'

        """dataset"""
        self.lcquad_test_dir = self.dataset + 'lcquad-1.0-test-data-new-nju_1209.json'
        self.lcquad_train_dir = self.dataset + 'lcquad-1.0-train-data-new-nju_1209.json'

        """bgp"""
        self.lcquad_train_bgp_dir = self.dataset + '2019.07.28_lcquad_train_bgps.txt'
        self.lcquad_test_bgp_dir = self.dataset + '2019.07.28_lcquad_test_bgps.txt'

        """node mention and answers"""
        self.lcquad_all_q_node_ann_dir = self.dataset+'FullyAnnotated_LCQuAD5000_node_mention_answers_nju.json'

        """oracle cache"""
        self.grounded_graph_file = self.dataset+'cache/oracle_grounded_graph_lcquad/'

        """semantic matching"""
        self.score12_match = self.dataset+'data_path_match_score12_e2e_ir_nomerge_0131/'


class GraphqFileName():

    def __init__(self, root):
        self.dataset = root + '/dataset_graphquestions/'

        """dataset"""
        self.graphquestions_testing_dir = self.dataset + 'graphquestions.testing_nju_1209.json'
        self.graphquestions_training_dir = self.dataset + 'graphquestions.training_nju_1209.json'

        """oracle cache"""
        self.cache_mid_to_names = self.dataset+'/cache_mid_to_names.json'
        self.grounded_graph_file = self.dataset+'cache/oracle_grounded_graph_graphq/'

        """node mention and answers"""
        self.graphquestions_node_ann_dir = self.dataset + 'FullyAnnotated_GraphQuestion5166_node_mention_nju_0410.json'

        """semantic matching"""
        self.score12_match = self.dataset+'data_path_match_score12_e2e_sp_v0.1_wo_agg/'


class CWQFileName():

    def __init__(self, root):
        self.dataset = root + '/dataset_cwq_1_1/'

        """dataset"""
        self.complexwebquestion_test_dir = self.dataset + 'ComplexWebQuestions_test_replaceOR_1209.json'
        self.complexwebquestion_train_dir = self.dataset + 'ComplexWebQuestions_train_replaceOR_1209.json'
        self.complexwebquestion_dev_dir = self.dataset + 'ComplexWebQuestions_dev_replaceOR_1209.json'

        """node mention and answers"""
        self.complexwebquestion_all_questions_dir = self.dataset + 'ComplexWebQuestions_1_0_all_question.json'

        """bgp"""
        self.complexwebquestion_test_bgp_dir = self.dataset +'ComplexWebQuestions_test_bgp.txt'
        self.complexwebquestion_train_bgp_dir = self.dataset +'ComplexWebQuestions_train_bgp.txt'
        self.complexwebquestion_dev_bgp_dir = self.dataset +'ComplexWebQuestions_dev_bgp.txt'

        """oracle cache"""
        self.grounded_graph_file = self.dataset+'cache/racle_grounded_graph_cwq/'
        self.cache_mid_to_names = self.dataset+'/cache_mid_to_names.json'

        """node mention and answers"""
        self.complexwebquestion_node_ann_dir = self.dataset + 'FullyAnnotated_ComplexWebQuestions_all_questions_node_mention_nju_0608.json'

        """semantic matching"""
        self.score12_match = self.dataset+'data_path_match_score12_e2e_ir_nomerge_0210_v0.2/'


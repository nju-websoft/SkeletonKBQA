import collections
from common.globals_args import fn_graph_file, fn_cwq_file, fn_lcquad_file
import pickle
import operator
from method_ir.grounding.path_match_score12 import score12_utils


class PathMatchScore12():

    def __init__(self, q_mode):
        self.q_mode = q_mode
        self.output_result_pickle = None
        self.set_model_data()

    # 1
    def set_model_data(self):
        if self.q_mode == 'cwq':
            model_file = fn_cwq_file.score12_match+'IR_5_v0.1_bert_slotptr_w_dep_cwq_epoch3_bs64_negs100_lr1e-05_structskeleton_4710_highest_test_result_4286.pickle'
            self.output_result_pickle = pickle.load(open(model_file, 'rb'))
        elif self.q_mode == 'lcquad':
            model_file = fn_lcquad_file.score12_match + "IR_5_bert_slotptr_w_dep_lcquad_epoch3_bs64_negs100_lr1e-05_structskeleton_44979_highest_test_result_627345.pickle"
            self.output_result_pickle = pickle.load(open(model_file, 'rb'))
        elif self.q_mode == 'graphq':
            model_file = fn_graph_file.score12_match+'IR_6_v0.1_bert_slotptr_w_dep_graphq_epoch3_bs16_negs100_lr1e-05_structdep_2894_highest_test_result_284956.pickle'
            self.output_result_pickle = pickle.load(open(model_file, 'rb'))

    # 2
    def _get_candidates_paths_with_scores(self, question_normal):
        candidates_paths_with_scores_list = []
        question_normal_log = {}
        for one in self.output_result_pickle['runtime']:
            for key in one['log']:
                if key == 'language_question_normal':
                    if one['log'][key] == question_normal:
                        question_normal_log = one['log']
        for key_in_log in question_normal_log:
            if key_in_log == 'candidate_score':
                for candidate, score in question_normal_log[key_in_log]:
                    candidates_paths_with_scores_list.append((candidate, score))
        return candidates_paths_with_scores_list

    # 3
    def set_bert_score_score12(self, question_normal, grounded_graph_forest_list):
        # a (['+', 'http://dbpedia.org/property/owner', '+', 'http://dbpedia.org/property/author'] -11.58642)
        candidates_paths_with_scores_list = self._get_candidates_paths_with_scores(question_normal=question_normal)
        # b grounded_graph_to_hops_to_candidates
        if self.q_mode == 'graphq':
            hops_grounded_graph_path_dict, hops_grounded_graph_id_dict = grounded_graphs_to_hops(grounded_graph_forest_list=grounded_graph_forest_list)
        elif self.q_mode == 'lcquad':
            hops_grounded_graph_path_dict, hops_grounded_graph_id_dict = grounded_graphs_to_hops(grounded_graph_forest_list=grounded_graph_forest_list)
        elif self.q_mode == 'cwq':
            hops_grounded_graph_path_dict, hops_grounded_graph_id_dict = grounded_graphs_to_hops(grounded_graph_forest_list=grounded_graph_forest_list)
        else:
            hops_grounded_graph_path_dict = dict()
            hops_grounded_graph_id_dict = dict()
        candidates_paths_using_grounded_graph_id = hops_to_candidates(hops_dict=hops_grounded_graph_id_dict)
        candidates_paths_using_grounded_graph_path = hops_to_candidates(hops_dict=hops_grounded_graph_path_dict)
        # c alignment, grounded_query_id -> candidate_path_index
        grounded_graph_id_to_candidates_path_index_dict = collections.OrderedDict()
        for path_index, candidate_path_grounded_graph_id in enumerate(candidates_paths_using_grounded_graph_id):
            grounded_graph_id_to_candidates_path_index_dict[candidate_path_grounded_graph_id] = path_index
        """test"""
        if len(candidates_paths_using_grounded_graph_id) != len(candidates_paths_with_scores_list):
            print(len(candidates_paths_using_grounded_graph_id), len(candidates_paths_with_scores_list))
        # j = 0
        for candidates_path, slbert_output_path in zip(candidates_paths_using_grounded_graph_path, candidates_paths_with_scores_list):
            # j += 1
            if not operator.eq(candidates_path, slbert_output_path[0]):
                print(candidates_path, '\t', slbert_output_path)
        # d get scores
        scores = []
        for grounded_graph in grounded_graph_forest_list:
            if grounded_graph.grounded_query_id in grounded_graph_id_to_candidates_path_index_dict:
                candidates_path_index = grounded_graph_id_to_candidates_path_index_dict[grounded_graph.grounded_query_id]
                score = candidates_paths_with_scores_list[candidates_path_index][-1]
            else:
                score = 0.0
            scores.append(score)
        return scores


def grounded_graphs_to_hops(grounded_graph_forest_list):
    hop1, hop2, hop3, hop4 = [],[],[],[]
    hop1_grounded_query_id, hop2_grounded_query_id, hop3_grounded_query_id, hop4_grounded_query_id = [], [], [], []
    for grounded_graph in grounded_graph_forest_list:
        grounded_query_id = grounded_graph.grounded_query_id
        triples = score12_utils.get_triples_by_grounded_graph_edges(nodes=grounded_graph.nodes, edges=grounded_graph.edges)
        has_uri_answer_node = False
        for triple in triples:
            if triple['subject'] == '?a' or triple['object'] == '?a':
                has_uri_answer_node = True
        if has_uri_answer_node:
            paths = score12_utils.triples_to_path_list(triples=triples, _root_id='?a')
        else:  # e1-to-e2
            entitys = []
            for triple in triples:
                if 'http://dbpedia.org/resource/' in triple['subject']:
                    entitys.append(triple['subject'])
                if 'http://dbpedia.org/resource/' in triple['object']:
                    entitys.append(triple['object'])
            entitys.sort()
            new_triples = score12_utils.rerank_triples(triples=triples)
            paths = score12_utils.triples_to_paths_lcquad_e1e2(triples=new_triples, entitys=entitys)
        if len(paths) == 1 * 2:
            hop1.append(paths)
            hop1_grounded_query_id.append(grounded_query_id)
        elif len(paths) == 2 * 2:
            hop2.append(paths)
            hop2_grounded_query_id.append(grounded_query_id)
        elif len(paths) == 3 * 2:
            hop3.append(paths)
            hop3_grounded_query_id.append(grounded_query_id)
        elif len(paths) == 4 * 2:
            hop4.append(paths)
            hop4_grounded_query_id.append(grounded_query_id)

    hops_dict = dict()
    hops_grounded_graph_id_dict = dict()
    if len(hop1) > 0:
        hops_dict['hop1'] = hop1
        hops_grounded_graph_id_dict['hop1'] = hop1_grounded_query_id
    if len(hop2) > 0:
        hops_dict['hop2'] = hop2
        hops_grounded_graph_id_dict['hop2'] = hop2_grounded_query_id
    if len(hop3) > 0:
        hops_dict['hop3'] = hop3
        hops_grounded_graph_id_dict['hop3'] = hop3_grounded_query_id
    if len(hop4) > 0:
        hops_dict['hop4'] = hop4
        hops_grounded_graph_id_dict['hop4'] = hop4_grounded_query_id
    return hops_dict, hops_grounded_graph_id_dict


def hops_to_candidates(hops_dict):
    candidates = []
    for key in ['hop4', 'hop3_2', 'hop3_1', 'hop3_0', 'hop3', 'hop2', 'hop1']:
        if key in hops_dict:
            candidates += hops_dict[key]
    return candidates


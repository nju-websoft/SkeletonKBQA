import os

from common.globals_args import fn_cwq_file
from common.hand_files import write_json, read_json, read_structure_file
from method_sp.grounding.grounding_args import property_reverse_dict
from skeleton_parsing import dep_to_path
from method_ir.grounding.path_match_score12 import score12_utils, triple_enum
from method_sp.parsing import parsing_utils
from datasets_interface import questions_utils

"""1 triple_path"""


def generate_cwq_gold_triple_path_annotation(cwq_annotation_list, qid_to_grounded_graph_dict, cwq_gold_nodes, output_file, is_deppath=False):
    from method_sp.parsing import node_recognition
    qid_graphq_nodes={}
    for one in cwq_gold_nodes:
        qid_graphq_nodes[one['qid']]=one
    print(len(cwq_annotation_list))
    for i in range(len(cwq_annotation_list)):
        graphq_struct_one=cwq_annotation_list[i]
        qid=graphq_struct_one.ID
        if qid not in qid_graphq_nodes:
            continue
        if qid not in qid_to_grounded_graph_dict:
            continue
        print(i, qid)
        if is_deppath:
            question_normal = graphq_struct_one.question
            '''gold'''
            # ungrounded_nodes = node_recognition.generate_gold_nodes(question_normal=question_normal)
            '''system'''
            tokens = parsing_utils.create_tokens(question_normal.split(" "))
            ungrounded_nodes = node_recognition.generate_nodes(question_normal=question_normal, qid=None, tokens=tokens)
            abstract_question_deppath_list = dep_to_path.get_deppath_list(question_normal=question_normal,
                                                                        ungrounded_nodes=ungrounded_nodes,
                                                                        isSkeletonorDep='Dep')
            qid_graphq_nodes[qid]['gold']['abstract_question_deppath'] = abstract_question_deppath_list
        gold_grounded_graph = qid_to_grounded_graph_dict[qid]
        triples_ = score12_utils.get_triples_by_grounded_graph_edges(nodes=gold_grounded_graph.nodes, edges=gold_grounded_graph.edges)
        reverse_triples_list = triple_enum.get_all_reverse_triples(triples=triples_, property_reverse_dict=property_reverse_dict)
        reverse_paths_list = []
        for index, reverse_triples in enumerate(reverse_triples_list):
            reverse_paths_list.append(score12_utils.triples_to_path_list(triples=reverse_triples, _root_id='?x'))
        qid_graphq_nodes[qid]['gold']['path'] = score12_utils.triples_to_path_list(triples=triples_, _root_id='?x')
        qid_graphq_nodes[qid]['gold']['reverse_paths_list'] = reverse_paths_list
    write_json(cwq_gold_nodes, fn_cwq_file.score12_match+output_file)


"""2 candidate path for training set"""


def generate_cwq_train_candidates_paths_from_structure(cwq_gold_path_list, train_candidates_sp_path_top_path, output_file):
    files = os.listdir(train_candidates_sp_path_top_path)
    new_cwq_path_list = []
    for one in cwq_gold_path_list:
        print(one['qid'])
        if str(one['qid'])+'.json' not in files:
            continue
        if 'path' not in one['gold']:
            continue
        new_one = dict()
        new_one['qid'] = one['qid']
        new_one['question_normal'] = one['question_normal']
        new_one['gold'] = one['gold']
        test_candidates_sp = read_structure_file(train_candidates_sp_path_top_path+str(one['qid'])+'.json')
        test_candidates_sp = test_candidates_sp[0]
        ungrounded_graph=test_candidates_sp.ungrounded_graph_forest[-1]
        hop1, hop2, hop3, hop4 = score12_utils.grounded_graph_list_to_path_list(ungrounded_graph.get_grounded_graph_forest())
        hops = []
        if len(hop1)>0:
            new_one['gold']['hop1']=hop1
            hops += hop1
        if len(hop2)>0:
            new_one['gold']['hop2']=hop2
            hops += hop2
        if len(hop3)>0:
            new_one['gold']['hop3']=hop3
            hops += hop3
        if len(hop4)>0:
            new_one['gold']['hop4']=hop4
            hops += hop4
        goldpath = None
        for hop in hops:
            for i, temp_goldpath in enumerate(new_one['gold']['reverse_paths_list']):
                if score12_utils.eq_paths(temp_goldpath, hop):
                    goldpath = temp_goldpath
                    break
        if goldpath is not None:
            new_one['gold']['path'] = goldpath
        del new_one['gold']['reverse_paths_list']
        new_cwq_path_list.append(new_one)
    write_json(new_cwq_path_list, fn_cwq_file.score12_match+output_file)


"""3 candidate path for testing set"""


def generate_cwq_test_e2e_candidate_paths_from_structure(cwq_gold_path_list, test_candidates_sp_path_top_path, output_file):

    def get_node(grounded_graph_pattern):
        nodes = []
        for node in grounded_graph_pattern.nodes:
            if node.node_type == 'entity':
                nodes.append({'tag': 'entity', 'uri': node.id})
            elif node.node_type == 'literal':
                nodes.append({'tag': 'literal', 'uri': node.id})
        return nodes

    def get_abstract_q(question_normal_, sequence_ner_tag_dict_):
        question_words = question_normal_.split()
        for key in sequence_ner_tag_dict_:
            if sequence_ner_tag_dict_[key] == 'entity':
                start, end = key.split('\t')
                start = int(start)
                end = int(end)
                question_words[start] = '<e>'
                for i in range(start + 1, end + 1):
                    question_words[i] = '$$$'
            elif sequence_ner_tag_dict_[key] == 'literal':
                start, end = key.split('\t')
                start = int(start)
                end = int(end)
                question_words[start] = '<l>'
                for i in range(start + 1, end + 1):
                    question_words[i] = '$$$'
        abstractquestion = ''
        for i, word in enumerate(question_words):
            if word != '$$$':
                abstractquestion += word
                if i < len(question_words) - 1:
                    abstractquestion += ' '
        return abstractquestion

    files = os.listdir(test_candidates_sp_path_top_path)
    new_cwq_gold_path_list = []
    count = 0
    for one in cwq_gold_path_list:
        qid = one['qid']
        print(qid)
        if str(one['qid']) + '.json' not in files:
            continue
        new_one = dict()
        new_one['qid'] = one['qid']
        new_one['question_normal'] = one['question_normal']
        new_one['gold'] = one['gold']
        question_normal = one['question_normal']
        test_candidates_sp = read_structure_file(test_candidates_sp_path_top_path + str(one['qid']) + '.json')
        ungrounded_graph = test_candidates_sp[0].ungrounded_graph_forest[-1]
        grounded_graph_forest = ungrounded_graph.get_grounded_graph_forest()
        sequence_ner_tag_dict = eval(ungrounded_graph.sequence_ner_tag_dict)
        new_one['pred'] = {'abstract_question': get_abstract_q(question_normal, sequence_ner_tag_dict),
                           'nodes': get_node(grounded_graph_pattern=grounded_graph_forest[0]),
                           'function': one['gold']['function']}
        hop1, hop2, hop3, hop4 = score12_utils.grounded_graph_list_to_path_list(ungrounded_graph.get_grounded_graph_forest())
        hops = []
        if len(hop1)>0:
            new_one['pred']['hop1']=hop1
            hops += hop1
        if len(hop2)>0:
            new_one['pred']['hop2']=hop2
            hops += hop2
        if len(hop3)>0:
            new_one['pred']['hop3']=hop3
            hops += hop3
        if len(hop4)>0:
            new_one['pred']['hop4']=hop4
            hops += hop4
        goldpath = None
        for hop in hops:
            for i, temp_goldpath in enumerate(new_one['gold']['reverse_paths_list']):
                if score12_utils.eq_paths(temp_goldpath, hop):
                    goldpath = temp_goldpath
                    count+=1
                    break
        if goldpath is not None:
            new_one['gold']['path'] = goldpath
        del new_one['gold']['reverse_paths_list']
        new_cwq_gold_path_list.append(new_one)
    write_json(new_cwq_gold_path_list, fn_cwq_file.score12_match+output_file)
    print(count)


if __name__ == '__main__':
    pass
    # 1 triples
    # dev
    """
    qid_to_grounded_graph_dict = questions_utils.extract_grounded_graph_from_jena_freebase(fn_cwq_file.complexwebquestion_dev_bgp_dir)
    dev_graph_nodes=read_json(fn_cwq_file.score12_match+'e2e_dev_cwq_gold_node_gold_path.json')
    generate_cwq_gold_triple_path_annotation(cwq_annotation_list=complexwebq_dev_list,
                                             qid_to_grounded_graph_dict=qid_to_grounded_graph_dict,
                                             cwq_gold_nodes=dev_graph_nodes,
                                             output_file="e2e_dev_cwq_gold_node_gold_path_dep_path.json",
                                             is_deppath=True)
    """

    # test
    """
    qid_to_grounded_graph_dict = questions_utils.extract_grounded_graph_from_jena_freebase(fn_cwq_file.complexwebquestion_test_bgp_dir)
    test_graph_nodes=read_json(fn_cwq_file.score12_match+'e2e_test_cwq_gold_node_gold_path.json')
    generate_cwq_gold_triple_path_annotation(cwq_annotation_list=complexwebq_test_list,
                                             qid_to_grounded_graph_dict=qid_to_grounded_graph_dict,
                                             cwq_gold_nodes=test_graph_nodes,
                                             output_file="e2e_test_cwq_gold_node_gold_path_dep_path.json",
                                             is_deppath=True)
    """

    # train
    """
    qid_to_grounded_graph_dict = questions_utils.extract_grounded_graph_from_jena_freebase(fn_cwq_file.complexwebquestion_train_bgp_dir)
    train_graph_nodes=read_json(fn_cwq_file.score12_match+'e2e_train_cwq_gold_node_gold_path.json')
    generate_cwq_gold_triple_path_annotation(cwq_annotation_list=complexwebq_train_list,
                                             qid_to_grounded_graph_dict=qid_to_grounded_graph_dict,
                                             cwq_gold_nodes=train_graph_nodes,
                                             output_file="e2e_train_cwq_gold_node_gold_path_dep_path.json",
                                             is_deppath=True)
    """

    # 2 train candidate path
    """
    train_graphq_path=read_json(fn_cwq_file.score12_match+'e2e_train_cwq_gold_node_gold_path_dep_path.json')
    train_candidates_sp_path_top_path=fn_cwq_file.dataset +'output_cwq_e2e/output_cwq_ir_skeleton_score12_ir5_v0.1_wo_agg/' \
                                                           '2.0_train_woagg_14353/'
    generate_cwq_train_candidates_paths_from_structure(cwq_gold_path_list=train_graphq_path,
                                                       train_candidates_sp_path_top_path=train_candidates_sp_path_top_path,
                                                       output_file="IR_6_v0.1_E2E_train_cwq_candidate_path_0227.json")
    """

    # 2 dev candidate path
    """
    dev_graphq_path = read_json(fn_cwq_file.score12_match+'e2e_dev_cwq_gold_node_gold_path_dep_path.json')
    dev_candidates_sp_path_top_path = fn_cwq_file.dataset +'output_cwq_e2e/output_cwq_ir_skeleton_score12_ir5_v0.1_wo_agg/' \
                                                         '2.0_dev_woagg_1748/'
    generate_cwq_train_candidates_paths_from_structure(cwq_gold_path_list=dev_graphq_path,
                                                       train_candidates_sp_path_top_path=dev_candidates_sp_path_top_path,
                                                       output_file="IR_6_v0.1_E2E_dev_cwq_candidate_path_0227.json")
    """

    # 3 test candidate path
    """
    test_graphq_path=read_json(fn_cwq_file.score12_match+'e2e_test_cwq_gold_node_gold_path_dep_path.json')
    test_candidates_sp_path_top_path=fn_cwq_file.dataset +'output_cwq_e2e/output_cwq_ir_skeleton_score12_ir5_v0.1_wo_agg/' \
                                                          '2.0_test_woagg_1774/'
    generate_cwq_test_e2e_candidate_paths_from_structure(cwq_gold_path_list=test_graphq_path,
                                                         test_candidates_sp_path_top_path=test_candidates_sp_path_top_path,
                                                         output_file="IR_6_v0.1_E2E_test_cwq_candidate_path_0227.json")
    """


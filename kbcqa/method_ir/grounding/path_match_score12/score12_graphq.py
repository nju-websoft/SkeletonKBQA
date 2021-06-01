from common.globals_args import fn_graph_file
from common.hand_files import write_json, read_json, read_structure_file
from datasets_interface.question_interface.graphquestion_interface import annotation_node_questions_json, test_graph_questions_struct, train_graph_questions_struct

from method_sp.grounding.grounding_args import property_reverse_dict
from skeleton_parsing import dep_to_path
import os
from method_ir.grounding.path_match_score12 import score12_utils, triple_enum

"""1 node and dependency path"""


def generate_graphq_gold_node_annotation_w_deppath(is_deppath=False):
    from method_sp.parsing import node_recognition
    train_result = []
    test_result = []
    questionnormal_function_abstract_question_nodes = dict()
    for one in annotation_node_questions_json:
        nodes = []
        for node in one['node_mention_nju']:
            if node['tag'] == 'entity' and node['uri'] is not None:
                nodes.append(node)
            elif node['tag'] == 'literal' and node['uri'] is not None:
                nodes.append(node)
        ann_dict = {
                'function': 'none',
                'abstract_question': one['abstract_question'],
                'nodes': nodes
        }
        if is_deppath:
            ungrounded_nodes = node_recognition.generate_gold_nodes(question_normal=one['question_normal'])
            abstract_question_deppath_list = dep_to_path.get_deppath_list(question_normal=one['question_normal'],
                                                                        ungrounded_nodes=ungrounded_nodes,
                                                                        isSkeletonorDep='Skeleton')
            ann_dict['abstract_question_deppath'] = abstract_question_deppath_list
        questionnormal_function_abstract_question_nodes[one['question_normal']] = ann_dict
    for one in train_graph_questions_struct:
        new_one = {}
        new_one['qid'] = one.qid
        new_one['question_normal'] = one.question
        if one.question in questionnormal_function_abstract_question_nodes:
            new_one['gold'] = questionnormal_function_abstract_question_nodes[one.question]
            new_one['gold']['function'] = one.function
            train_result.append(new_one)
    for one in test_graph_questions_struct:
        new_one = {}
        new_one['qid'] = one.qid
        new_one['question_normal'] = one.question
        if new_one['question_normal'] in questionnormal_function_abstract_question_nodes:
            new_one['gold'] = questionnormal_function_abstract_question_nodes[one.question]
            new_one['gold']['function'] = one.function
            test_result.append(new_one)
    write_json(train_result, fn_graph_file.score12_match + "train_graphq_gold_node_0124.json")
    write_json(test_result, fn_graph_file.score12_match + "test_graphq_gold_node_0124.json")


"""2 triple_path"""


def generate_graphq_gold_triple_path_annotation(graphq_annotation_list, graphq_gold_nodes, output_file):
    qid_graphq_nodes={}
    for one in graphq_gold_nodes:
        qid_graphq_nodes[one['qid']]=one
    for i in range(len(graphq_annotation_list)):
        graphq_struct_one=graphq_annotation_list[i]
        qid = graphq_struct_one.qid
        print(qid)
        if qid not in qid_graphq_nodes:
            continue
        triples_ = score12_utils.get_triples_by_grounded_graph_edges_graphq(nodes=graphq_struct_one.nodes, edges=graphq_struct_one.edges)
        reverse_triples_list = triple_enum.get_all_reverse_triples(triples=triples_, property_reverse_dict=property_reverse_dict)
        reverse_paths_list = []
        for reverse_triples in reverse_triples_list:
            reverse_paths_list.append(score12_utils.triples_to_path_list(triples=reverse_triples, _root_id='?x'))
        qid_graphq_nodes[qid]['gold']['path'] = score12_utils.triples_to_path_list(triples=triples_, _root_id='?x')
        qid_graphq_nodes[qid]['gold']['reverse_paths_list'] = reverse_paths_list
    write_json(graphq_gold_nodes, fn_graph_file.score12_match+output_file)


"""3 candidate path for training set"""


def generate_graphq_train_candidates_paths_from_structure(graphq_gold_path_list, train_candidates_sp_path_top_path, output_file):
    files = os.listdir(train_candidates_sp_path_top_path)
    for one in graphq_gold_path_list:
        print(one['qid'])
        if str(one['qid'])+'.json' not in files:
            continue
        if 'path' not in one['gold']:
            continue
        test_candidates_sp = read_structure_file(train_candidates_sp_path_top_path+str(one['qid'])+'.json')
        test_candidates_sp = test_candidates_sp[0]
        ungrounded_graph=test_candidates_sp.ungrounded_graph_forest[-1]
        hop1, hop2, hop3, hop4 = score12_utils.grounded_graph_list_to_path_list(ungrounded_graph.get_grounded_graph_forest())
        hops = []
        if len(hop1)>0:
            one['gold']['hop1']=hop1
            hops += hop1
        if len(hop2)>0:
            one['gold']['hop2']=hop2
            hops += hop2
        if len(hop3)>0:
            one['gold']['hop3']=hop3
            hops += hop3
        if len(hop4)>0:
            one['gold']['hop4']=hop4
            hops += hop4
        goldpath = None
        for hop in hops:
            for i, temp_goldpath in enumerate(one['gold']['reverse_paths_list']):
                if score12_utils.eq_paths(temp_goldpath, hop):
                    goldpath = temp_goldpath
                    break
        if goldpath is not None:
            one['gold']['path'] = goldpath
        del one['gold']['reverse_paths_list']
    write_json(graphq_gold_path_list, fn_graph_file.score12_match+output_file)


"""4 candidate path for testing set"""


def generate_graphq_test_e2e_candidate_paths_from_structure(graphq_gold_path_list, test_candidates_sp_path_top_path, output_file):

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
    for one in graphq_gold_path_list:
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
    write_json(new_cwq_gold_path_list, fn_graph_file.score12_match+output_file)
    print(count)


def ir_interface():
    # 1 node
    generate_graphq_gold_node_annotation_w_deppath(is_deppath=True)

    # 2 triples
    """
    test_graph_nodes = read_json(fn_graph_file.score12_match + 'dep_test_graphq_gold_node_1331.json')
    generate_graphq_gold_triple_path_annotation(
        graphq_annotation_list=test_graph_questions_struct,
        graphq_gold_nodes=test_graph_nodes,
        output_file="dep_test_graphq_wskeletonpath_1331_gold_path.json")
    train_graph_nodes=read_json(fn_graph_file.score12_match+'dep_train_graphq_gold_node_1331.json')
    generate_graphq_gold_triple_path_annotation(
                                           graphq_annotation_list=train_graph_questions_struct,
                                           graphq_gold_nodes=train_graph_nodes,
                                           output_file="dep_train_graphq_wskeletonpath_1331_gold_path.json")
    """

    # 3 train candidate path
    """
    train_graphq_path=read_json(fn_graph_file.score12_match+'dep_train_graphq_wskeletonpath_1231_gold_path.json')
    train_candidates_sp_path_top_path=fn_graph_file.dataset +'output_graphq_e2e/output_graphq_ir_skeleton_score12_ir5_v0.1_wo_agg/' \
                                                             '2.0_train_1708/'
    generate_graphq_train_candidates_paths_from_structure(graphq_gold_path_list=train_graphq_path,
                                                       train_candidates_sp_path_top_path=train_candidates_sp_path_top_path,
                                                       output_file="IR_6_v0.1_E2E_dep_train_graphq_candidate_path_0225.json")
    """

    # 4 test candidate path
    """
    test_graphq_path=read_json(fn_graph_file.score12_match+'dep_test_graphq_wskeletonpath_1231_gold_path.json')
    test_candidates_sp_path_top_path=fn_graph_file.dataset +'output_graphq_e2e/output_graphq_ir_dep_score12_ir6_v0.1_wo_agg/' \
                                                            '2.0_test_1149/'
    generate_graphq_test_e2e_candidate_paths_from_structure(graphq_gold_path_list=test_graphq_path,
                                                        test_candidates_sp_path_top_path=test_candidates_sp_path_top_path,
                                                        output_file="IR_6_v0.1_E2E_dep_test_graphq_candidate_path_0225.json")
    """


def sp_interface():
    # 1 node
    generate_graphq_gold_node_annotation_w_deppath(is_deppath=False)

    # 2 triples
    """
    test_graph_nodes = read_json(fn_graph_file.score12_match + 'test_graphq_gold_node_0124.json')
    generate_graphq_gold_triple_path_annotation(
        graphq_annotation_list=test_graph_questions_struct,
        graphq_gold_nodes=test_graph_nodes,
        output_file="test_graphq_gold_node_gold_path_0124.json")
    train_graph_nodes=read_json(fn_graph_file.score12_match+'train_graphq_gold_node_0124.json')
    generate_graphq_gold_triple_path_annotation(
                                           graphq_annotation_list=train_graph_questions_struct,
                                           graphq_gold_nodes=train_graph_nodes,
                                           output_file="train_graphq_gold_node_gold_path_0124.json")
    """

    # 3 train candidate path
    """
    train_graphq_path=read_json(fn_graph_file.score12_match+'train_graphq_gold_node_gold_path_0124.json')
    train_candidates_sp_path_top_path=fn_graph_file.dataset +'output_graphq_e2e/output_graphq_sp_dep_slot_e2e_sp4_完成/' \
                                                             '2.2_train_merge_agg_1518/'
    generate_graphq_train_candidates_paths_from_structure(graphq_gold_path_list=train_graphq_path,
                                                       train_candidates_sp_path_top_path=train_candidates_sp_path_top_path,
                                                       output_file="SP_4_E2E_train_graphq_candidate_path_0223_all_v0.3.json")
    """

    # 4 test candidate path
    """
    test_graphq_path = read_json(fn_graph_file.score12_match+'test_graphq_gold_node_gold_path_0124.json')
    test_candidates_sp_path_top_path = fn_graph_file.dataset +'output_graphq_e2e/output_graphq_sp_skeleton_slot_e2e_sp2_完成/' \
                                                              '2.2_test_merge_agg_37_989/'
    generate_graphq_test_e2e_candidate_paths_from_structure(graphq_gold_path_list=test_graphq_path,
                                                        test_candidates_sp_path_top_path=test_candidates_sp_path_top_path,
                                                        output_file="SP_2_E2E_test_graphq_candidate_path_0220_v0.2.json")
    """


if __name__ == '__main__':
    ir_interface()
    # sp_interface()


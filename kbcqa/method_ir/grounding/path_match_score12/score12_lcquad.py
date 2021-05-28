from common.globals_args import fn_lcquad_file
from datasets_interface.question_interface.lcquad_1_0_interface import annotation_node_answers_all_questions_json, lcquad_train_list, lcquad_test_list
from common.hand_files import write_json, read_structure_file, read_json
from method_ir.grounding.path_match_score12 import score12_utils
from skeleton_parsing import dep_to_path
import os
from method_sp.parsing import parsing_utils


"""1"""


def generate_lcquad_gold_node_annotation(is_deppath=False):
    '''
     {
         "qid": "655",
         "question_normal": "List some leaders of regions in the Indian Standard Time Zone ?",
         "gold": {
             "abstract_question": "List some leaders of regions in the <e> Zone ?",
             "nodes": [
                 {
                     "mention": "Indian Standard Time",
                     "start_index": 7,
                     "end_index": 9,
                     "tag": "entity",
                     "uri": "http://dbpedia.org/resource/Indian_Standard_Time"
                 }
             ],
             "function": "bgp"
         }
     }
     '''
    from method_sp.parsing import node_recognition
    train_result=[]
    test_result=[]
    questionnormal_function_abstract_question_nodes = dict()
    for index, one in enumerate(annotation_node_answers_all_questions_json):
        print(index, one['question_normal'])
        nodes = []
        for node in one['node_mention']:
            if node['tag'] == 'entity' and node['uri'] is not None:
                nodes.append(node)
        ann_dict = {
                'function': one['type'],
                'abstract_question': one['abstract_question'],
                'nodes': nodes
        }
        if is_deppath:
            '''gold'''
            # ungrounded_nodes = node_recognition.generate_gold_nodes(question_normal=one['question_normal'])
            '''system'''
            tokens = parsing_utils.create_tokens(one['question_normal'].split(" "))
            ungrounded_nodes = node_recognition.generate_nodes(question_normal=one['question_normal'], qid=None, tokens=tokens)
            abstract_question_deppath_list = dep_to_path.get_deppath_list(question_normal=one['question_normal'],
                                                                          ungrounded_nodes=ungrounded_nodes,
                                                                          isSkeletonorDep='Skeleton') #Skeleton Dep
            ann_dict['abstract_question_deppath'] = abstract_question_deppath_list
        questionnormal_function_abstract_question_nodes[one['question_normal']] = ann_dict

    for one in lcquad_train_list:
        new_one={}
        new_one['qid']=one.qid
        new_one['question_normal']=one.question_normal
        if one.question_normal in questionnormal_function_abstract_question_nodes:
            new_one['gold'] = questionnormal_function_abstract_question_nodes[one.question_normal]
            train_result.append(new_one)
    for one in lcquad_test_list:
        new_one={}
        new_one['qid'] = one.qid
        new_one['question_normal'] = one.question_normal
        if new_one['question_normal'] in questionnormal_function_abstract_question_nodes:
            new_one['gold'] = questionnormal_function_abstract_question_nodes[one.question_normal]
            test_result.append(new_one)

    write_json(train_result, fn_lcquad_file.score12_match + "skeletonpath_train_lcquad_gold_node.json")
    write_json(test_result, fn_lcquad_file.score12_match + "skeletonpath_test_lcquad_gold_node.json")


"""2"""


def generate_lcquad_gold_triple_path_annotation(lcquad_annotation_list, lcquad_gold_nodes, output_file):
    '''
    "triples":
        [
            {
                "subject": "http://dbpedia.org/resource/Marine_Corps_Air_Station_Kaneohe_Bay",
                "predicate": "http://dbpedia.org/property/architect",
                "object": "?uri"
            },
            {
                "subject": "http://dbpedia.org/resource/New_Sanno_Hotel",
                "predicate": "http://dbpedia.org/ontology/tenant",
                "object": "?uri"
            }
        ]

    "path": [
    "+",
    "http://dbpedia.org/property/architect",
    "+",
    "http://dbpedia.org/ontology/tenant"
    ]
    '''

    qid_lcquad_nodes={}
    for one in lcquad_gold_nodes:
        qid_lcquad_nodes[one['qid']]=one
    for i in range(len(lcquad_annotation_list)):
        lcquad_struct_one=lcquad_annotation_list[i]
        qid=lcquad_struct_one.qid
        print(qid)
        if qid not in qid_lcquad_nodes:
            continue
        lcquad_nodes_one = qid_lcquad_nodes[qid]
        triples_ = lcquad_struct_one.parsed_sparql['where'][0]['triples']
        new_triples = []
        has_uri_answer_node = False
        entitys = []
        for triple in triples_:
            if triple['predicate']!='http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
                new_triples.append(triple)
            if triple['subject'] == '?uri' or triple['object'] == '?uri':
                has_uri_answer_node = True
            if 'http://dbpedia.org/resource/' in triple['subject']:
                entitys.append(triple['subject'])
            if 'http://dbpedia.org/resource/' in triple['object']:
                entitys.append(triple['object'])
        if has_uri_answer_node:
            lcquad_nodes_one['gold']['path'] = score12_utils.triples_to_path_list(triples=new_triples, _root_id='?uri')
        else:
            """is e1-to-e2"""
            entitys.sort()
            print(qid, entitys)
            new_triples = score12_utils.rerank_triples(triples=new_triples)
            lcquad_nodes_one['gold']['path'] = score12_utils.triples_to_paths_lcquad_e1e2(triples=new_triples, entitys=entitys)
    write_json(lcquad_gold_nodes, fn_lcquad_file.score12_match+output_file)


"""3 candidate path for training set"""


def generate_lcquad_train_candidates_paths_from_structure(lcquad_gold_path_list, train_candidates_sp_path_top_path, output_file):
    files = os.listdir(train_candidates_sp_path_top_path)
    for one in lcquad_gold_path_list:
        print(one['qid'])
        if str(one['qid'])+'.json' not in files:
            continue
        if 'path' not in one['gold']:
            continue
        test_candidates_sp = read_structure_file(train_candidates_sp_path_top_path+str(one['qid'])+'.json')
        test_candidates_sp = test_candidates_sp[0]
        ungrounded_graph=test_candidates_sp.ungrounded_graph_forest[-1]
        hop1, hop2, hop3, hop4 = score12_utils.grounded_graph_list_to_path_list(ungrounded_graph.get_grounded_graph_forest())
        if len(hop1)>0:
            one['gold']['hop1']=hop1
        if len(hop2)>0:
            one['gold']['hop2']=hop2
        if len(hop3)>0:
            one['gold']['hop3']=hop3
        if len(hop4)>0:
            one['gold']['hop4']=hop4
    write_json(lcquad_gold_path_list, fn_lcquad_file.score12_match+output_file)


"""4 candidate path for testing set"""


def generate_lcquad_test_e2e_candidate_paths_from_structure(lcquad_gold_path_list, test_candidates_sp_path_top_path, output_file):

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
    for one in lcquad_gold_path_list:
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
        new_cwq_gold_path_list.append(new_one)
    write_json(new_cwq_gold_path_list, fn_lcquad_file.score12_match+output_file)


def ir_interface():
    #1 node
    generate_lcquad_gold_node_annotation(is_deppath=False)

    #2 triple
    """
    train_lcquad_nodes=read_json(fn_lcquad_file.score12_match+'skeletonpath_train_lcquad_gold_node.json')
    generate_lcquad_gold_triple_path_annotation(lcquad_annotation_list=lcquad_train_list,
                                           lcquad_gold_nodes=train_lcquad_nodes,
                                           output_file="skeletonpath_train_lcquad_wgold_path.json")

    test_lcquad_nodes=read_json(fn_lcquad_file.score12_match+'skeletonpath_test_lcquad_gold_node.json')
    generate_lcquad_gold_triple_path_annotation(lcquad_annotation_list=lcquad_test_list,
                                           lcquad_gold_nodes=test_lcquad_nodes,
                                           output_file="skeletonpath_test_lcquad_gold_path.json")
    """

    # 3 train candidate path
    """
    train_graphq_path=read_json(fn_lcquad_file.score12_match+'deppath_train_lcquad_gold_path.json')
    train_candidates_sp_path_top_path=fn_lcquad_file.dataset +'output_lcquad_e2e/' \
                                                              'output_lcquad_ir_skeleton_score12_ir5/2.0_train_2703/'
    generate_lcquad_train_candidates_paths_from_structure(lcquad_gold_path_list=train_graphq_path,
                                                       train_candidates_sp_path_top_path=train_candidates_sp_path_top_path,
                                                       output_file="IR_6_E2E_train_dep_lcquad_candidate_path_0201.json")
    """

    # 4 test candidate path
    """
    test_graphq_path=read_json(fn_lcquad_file.score12_match+'deppath_test_lcquad_gold_path.json')
    test_candidates_sp_path_top_path=fn_lcquad_file.dataset +'output_lcquad_e2e/' \
                                                             'output_lcquad_ir_dep_score12_ir6/2.0_test_669/'
    generate_lcquad_test_e2e_candidate_paths_from_structure(lcquad_gold_path_list=test_graphq_path,
                                                            test_candidates_sp_path_top_path=test_candidates_sp_path_top_path,
                                                            output_file="IR_6_E2E_test_dep_lcquad_candidate_path_0201.json")
    """


def sp_interface():
    #1 node
    generate_lcquad_gold_node_annotation(is_deppath=False)

    #2 triple
    """
    test_lcquad_nodes=read_json(fn_lcquad_file.score12_match+'dep_el_test_lcquad_gold_node.json')
    generate_lcquad_gold_triple_path_annotation(lcquad_annotation_list=lcquad_test_list,
                                           lcquad_gold_nodes=test_lcquad_nodes,
                                           output_file="sp4_el_test_lcquad_gold_node_wgold_path.json")
    train_lcquad_nodes=read_json(fn_lcquad_file.score12_match+'dep_el_train_lcquad_gold_node.json')
    generate_lcquad_gold_triple_path_annotation(lcquad_annotation_list=lcquad_train_list,
                                           lcquad_gold_nodes=train_lcquad_nodes,
                                           output_file="sp4_el_train_lcquad_gold_node_wgold_path.json")
    """

    # 3 train candidate path
    """
    train_graphq_path=read_json(fn_lcquad_file.score12_match+'train_lcquad_gold_node_wgold_path_0131.json')
    train_candidates_sp_path_top_path=fn_lcquad_file.dataset +'output_lcquad_e2e/' \
                                                              'output_lcquad_sp_dep_slot_e2e_sp4/' \
                                                              '2.2_train_2149/'
    generate_lcquad_train_candidates_paths_from_structure(lcquad_gold_path_list=train_graphq_path,
                                                       train_candidates_sp_path_top_path=train_candidates_sp_path_top_path,
                                                       output_file="SP_4_E2E_train_lcquad_candidate_path_0201.json")
    """

    # 4 test candidate path
    """
    test_graphq_path=read_json(fn_lcquad_file.score12_match+'test_lcquad_gold_node_wgold_path_0131.json')
    test_candidates_sp_path_top_path=fn_lcquad_file.dataset +'output_lcquad_e2e/' \
                                                             'output_lcquad_sp_dep_slot_e2e_sp4/' \
                                                             '2.2_test_529/'
    generate_lcquad_test_e2e_candidate_paths_from_structure(lcquad_gold_path_list=test_graphq_path,
                                                            test_candidates_sp_path_top_path=test_candidates_sp_path_top_path,
                                                            output_file="SP_4_E2E_test_lcquad_candidate_path_0201.json")
    """


if __name__ == '__main__':
    ir_interface()
    # sp_interface()



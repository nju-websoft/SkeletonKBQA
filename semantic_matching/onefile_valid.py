import numpy
import numpy as np
import embeddings_interface
import torch
import data_loader as dl
import network as net
from configs.config_loader import Config
import os
import tensor_utils


class QuestionAnswering:
    """
        Usage:
            qa = QuestionAnswering(parameter_dict, False, _word_to_id, device, True)
            q = np.rancorechainmodeldom.randint(0, 1233, (542))
            p = np.random.randint(0, 123, (10, 55))
            print(qa._predict_corechain(q,p))
    """
    def __init__(self, parameters, device, _dataset, max_num_relations, debug, corechain_model, model_save_location=None):
        self.parameters = parameters
        self.debug = debug
        self.device = device
        self.parameters['dataset'] = _dataset
        self.max_num_relations=max_num_relations
        if corechain_model is None:
            self._load_corechain_model(model_save_location)
        else:
            self.corechain_model = corechain_model
        self.parameters['bidirectional'] = True


    def _load_corechain_model(self, model_save_location):
        m = self.parameters['corechainmodel']
        if m == 'bert_slotptr':
            print('bert_slotptr')
            self.corechain_model = net.Bert_Scorer_slotptr(_parameter_dict=self.parameters,  _device=self.device,max_num_relations=self.max_num_relations, _debug=self.debug)
        elif m == 'bert_slotptr_w_dep':
            print('bert_slotptr_w_dep')
            self.corechain_model = net.Bert_Scorer_slotptr_w_dep(_parameter_dict=self.parameters, _device=self.device, max_num_relations=self.max_num_relations, _debug=self.debug)
        elif m == 'bert_deppath':
            print('bert_deppath')
            self.corechain_model = net.Bert_DepMatch(_parameter_dict=self.parameters, _device=self.device, max_num_relations=self.max_num_relations, _debug=self.debug)
        elif m == 'slotptr':
            config=Config()
            self.corechain_model = net.QelosSlotPointerModel(_parameter_dict=self.parameters, _device=self.device, max_num_relations=self.max_num_relations, _debug=self.debug,config=config)
        elif m == 'bert':
            self.corechain_model = net.Bert_Scorer(_parameter_dict=self.parameters, _device=self.device, _debug=self.debug)
        if model_save_location is not None:
            model_path = model_save_location+'/model.torch'
        else:
            model_path = os.path.join(self.parameters['_model_dir'], 'core_chain')
            model_path = os.path.join(model_path, self.parameters['corechainmodel'])
            model_path = os.path.join(model_path, self.parameters['dataset'])
            model_path = os.path.join(model_path, self.parameters['corechainmodelnumber'])
            model_path = os.path.join(model_path, 'model.torch')
        self.corechain_model.load_from(model_path)
        self.parameters['corechainmodel'] = m


    # 3
    def _predict_corechain(self, _q, _question_dep, _question_dep_mask_matrix, _p, _p_words, _p1=None, _p2=None, _p3=None, _p4=None):
        """
            Given a datapoint (question, paths) encoded in  embedding_vocab, run the model's predict and find the best corechain.
            _q: (<var len>)
            _p: (100/500, <var len>)
            returns score: (100/500)
        """
        def distribute_it(np_array, k): # print(len(np_array))
            return np.array_split(np_array[:], k, axis=0)
        ###Pad questions
        Q = np.zeros((len(_p), self.parameters['max_length']))
        Q[:, :min(len(_q), self.parameters['max_length'])] = np.repeat(_q[np.newaxis, :min(len(_q), self.parameters['max_length'])], repeats=len(_p), axis=0)
        ###Pad question dep
        DEP = np.zeros((len(_p), len(_question_dep), self.parameters['max_length']))
        for _question_dep_index in range(len(_question_dep)):
            _question_dep_index_temp = np.asarray(_question_dep[_question_dep_index])
            DEP[:,_question_dep_index,:min(len(_question_dep[_question_dep_index]), self.parameters['max_length'])] = np.repeat(
                _question_dep_index_temp[np.newaxis, :min(len(_question_dep_index_temp), self.parameters['max_length'])], repeats=len(_p), axis=0
            )
        # DEP[:, :min(len(_question_dep), self.parameters['max_length'])] = np.repeat(_question_dep[np.newaxis, :min(len(_question_dep),self.parameters['max_length'])], repeats=len(_p), axis=0)
        DEP_MASK = np.zeros((len(_p), len(_question_dep_mask_matrix[0])))
        DEP_MASK[:,:min(len(_question_dep_mask_matrix[0]), self.parameters['max_length'])] = np.repeat(
            _question_dep_mask_matrix[:, :min(len(_question_dep_mask_matrix[0]), self.parameters['max_length'])], repeats=len(_p), axis=0
        )

        ### Pad paths
        P = np.zeros((len(_p), self.parameters['max_length']))
        P_words = np.zeros((len(_p), self.parameters['max_length']))
        if _p1:
            P1 = np.zeros((len(_p), self.parameters['max_length']))
            P2 = np.zeros((len(_p), self.parameters['max_length']))
            P3 = np.zeros((len(_p), self.parameters['max_length']))
            P4 = np.zeros((len(_p), self.parameters['max_length']))
        for i in range(len(_p)):
            P[i, :min(len(_p[i]), self.parameters['max_length'])] = _p[i][ :min(len(_p[i]), self.parameters['max_length'])]
            P_words[i, :min(len(_p_words[i]), self.parameters['max_length'])] = _p_words[i][ :min(len(_p_words[i]), self.parameters['max_length'])]
        if _p1:
            for i in range(len(_p)): # print(type(_p1[i]),_p1[i],_p1[:5])
                P1[i, :min(len(_p1[i]), self.parameters['max_length'])] = _p1[i][:min(len(_p1[i]),self.parameters['max_length'])]
                P2[i, :min(len(_p2[i]), self.parameters['max_length'])] = _p2[i][ :min(len(_p2[i]),self.parameters['max_length'])]
                P3[i, :min(len(_p3[i]), self.parameters['max_length'])] = _p3[i][ :min(len(_p3[i]),self.parameters['max_length'])]
                P4[i, :min(len(_p4[i]), self.parameters['max_length'])] = _p4[i][ :min(len(_p4[i]),self.parameters['max_length'])]
            if self.parameters['corechainmodel'] == 'slotptr' or \
                    self.parameters['corechainmodel'] == 'bert_slotptr' \
                    or self.parameters['corechainmodel'] == 'bert' \
                    or self.parameters['corechainmodel'] == 'bert_slotptr_w_dep'\
                    or self.parameters['corechainmodel'] == 'bert_deppath':
                P1 = P1[:, :self.parameters['relsp_pad']]
                P2 = P2[:, :self.parameters['relsp_pad']]
                P3 = P3[:, :self.parameters['relsp_pad']]
                P4 = P4[:, :self.parameters['relsp_pad']]
                # P1 = torch.tensor(P1, dtype=torch.long, device=self.device)
        # Convert np to torch stuff
        P = P[:, :self.parameters['rel_pad']]
        P_words = P_words[:, :self.parameters['rel_pad']]
        if not _p1: # Check what variables are None and which are not none.
            P1, P2, P3, P4 = None, None, None,None
        ########
        distribute = True
        k = 100
        if len(Q) < k + 1:
            distribute = False
        if distribute:  # len(Q) >= k + 1
            print("in distributed setting")
            if _p1:
                Q_dist, DEP_dist, DEP_MASK_dist, P_dist, P_words_dist, \
                P1_dist, P2_dist, P3_dist, P4_dist = distribute_it(Q, k), distribute_it(DEP, k), distribute_it(DEP_MASK, k), distribute_it(P, k), distribute_it(P_words, k),\
                                                      distribute_it(P1, k), distribute_it(P2, k), distribute_it(P3, k), distribute_it(P4, k)
                temp_score = []
                for q, dep, dep_mask, p,p_words, p1, p2, p3, p4 in zip(Q_dist, DEP_dist, DEP_MASK_dist, P_dist,P_words_dist, P1_dist, P2_dist,P3_dist,P4_dist):
                    if len(q)==1:
                        temp_score.append(self._tensorized_Score(q, dep, dep_mask, p, p_words, p1, p2, p3, p4))
                    else:
                        temp_score.append(self._tensorized_Score(q, dep, dep_mask, p, p_words, p1, p2, p3, p4))
            ###combine score
            final_score = []
            for scores in temp_score:
                for s in scores:
                    final_score.append(s)
            return np.asarray(final_score)
        else:
            return self._tensorized_Score(Q, DEP, DEP_MASK, P, P_words=P_words, P1=P1,  P2=P2, P3=P3, P4=P4)

    # 4
    def _tensorized_Score(self, Q, DEP, DEP_MASK, P, P_words,P1=None,P2=None, P3=None,P4=None):
        ###covernt tensor
        Q = torch.tensor(Q, dtype=torch.long, device=self.device)
        P = torch.tensor(P, dtype=torch.long, device=self.device)
        P_words = torch.tensor(P_words, dtype=torch.long, device=self.device)
        if type(P1) != type(None): # Then P2 also exists
            P1 = torch.tensor(P1, dtype=torch.long, device=self.device)
            P2 = torch.tensor(P2, dtype=torch.long, device=self.device)
            P3 = torch.tensor(P3, dtype=torch.long, device=self.device)
            P4 = torch.tensor(P4, dtype=torch.long, device=self.device)
        ###predict model
        if self.parameters['corechainmodel'] in ['slotptr', 'bert_slotptr']:
            score = self.corechain_model.predict(question=Q, paths=P,paths_words=P_words, paths_rel1=P1, paths_rel2=P2, paths_rel3=P3,paths_rel4=P4, device=self.device)
        elif self.parameters['corechainmodel']=='bert':
            score = self.corechain_model.predict(question=Q, paths=P, paths_words=P_words, device=self.device)
        elif self.parameters['corechainmodel'] in ['bert_slotptr_w_dep','bert_deppath']:
            DEP = torch.tensor(DEP, dtype=torch.long, device=self.device)
            DEP_MASK = torch.tensor(DEP_MASK, dtype=torch.float32, device=self.device)
            score = self.corechain_model.predict(question=Q, question_dep=DEP, question_dep_mask=DEP_MASK, paths=P, paths_words=P_words, paths_rel1=P1, paths_rel2=P2,paths_rel3=P3,paths_rel4=P4, device=self.device)
        if self.parameters['corechainmodel']=='bert':
            score=score.squeeze(1)
        return score.detach().cpu().numpy()


# 2.5
def _create_rd_sp_paths(paths,no_reldet=False):
    special_char = [embeddings_interface.vocabularize(['+']), embeddings_interface.vocabularize(['-'])]
    dummy_path = [0]
    paths_rel1_sp = []
    paths_rel2_sp = []
    paths_rel3_sp = []
    paths_rel4_sp = []
    for p in paths:
        p1, p2,p3,p4 = dl.break_path(p, special_char)
        paths_rel1_sp.append(p1)
        if p2 is not None:
            paths_rel2_sp.append(p2)
        else:
            paths_rel2_sp.append(dummy_path)
        if p3 is not None:
            paths_rel3_sp.append(p3)
        else:
            paths_rel3_sp.append(dummy_path)
        if p4 is not None:
            paths_rel4_sp.append(p4)
        else:
            paths_rel4_sp.append(dummy_path)
    paths_rel1_sp = [np.asarray(o) for o in paths_rel1_sp]
    paths_rel2_sp = [np.asarray(o) for o in paths_rel2_sp]
    paths_rel3_sp = [np.asarray(o) for o in paths_rel3_sp]
    paths_rel4_sp = [np.asarray(o) for o in paths_rel4_sp]
    return paths_rel1_sp,paths_rel2_sp,paths_rel3_sp,paths_rel4_sp


# 2
def corechain_prediction(question, question_dep, question_dep_mask_matrix, paths, paths_words,positive_path_index, no_positive_path,model,dataset, quesans, candidates):
    mrr = 0
    best_path_index=-2
    path_predicted_correct = False
    output=[]
    if no_positive_path and len(paths) == 0: ###There exists no positive path and also no negative paths
        print("The code should not have been herr. There is no warning.")
    elif not no_positive_path and len(paths) == 1: #There exists a positive path and there exists no negative path
        mrr = 1
        path_predicted_correct = True
        output=[1.0]
        best_path_index=0
    elif len(paths) >0:
        ###因为我的no_positive_path要求了要整个完全一样，而之前的lcquad和qald他没有这么要求，它只要最后的pathwords一样就行了
        ###path = positive_path + negative_paths
        if model in ['slotptr','bert_slotptr','bert','bert_slotptr_w_dep','bert_deppath']:
            paths_rel1_sp, paths_rel2_sp,paths_rel3_sp, paths_rel4_sp = _create_rd_sp_paths(paths,no_reldet=True)
            output = quesans._predict_corechain(question, question_dep, question_dep_mask_matrix, paths, paths_words, paths_rel1_sp, paths_rel2_sp,paths_rel3_sp,paths_rel4_sp)
        ###best index
        best_path_index = np.argmax(output)
        ###scoring
        if dataset in ['lcquad']:
            if np.array_equal(paths[best_path_index], paths[positive_path_index]) or tensor_utils.is_eq_twopaths(candidates[best_path_index], candidates[positive_path_index]):
                path_predicted_correct = True
                mrr=1.0
            else:
                if positive_path_index >= 0:
                    mrr_output = np.argsort(output)[::-1]  #x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
                    mrr_output = mrr_output.tolist()
                    mrr = mrr_output.index(positive_path_index) + 1.0  #position

        elif dataset in['graphq','complexwebq', 'cwq']:
            if best_path_index == positive_path_index:
                path_predicted_correct = True

            if positive_path_index >= 0:
                mrr_output = np.argsort(output)[::-1]
                mrr_output = mrr_output.tolist()
                mrr = mrr_output.index(positive_path_index) + 1.0
        else:
            raise IOError('dataset not in my scope!!!!')

        if mrr != 0:
            mrr = 1.0 / mrr
    else:
        print("The code should not have been herr.")
        raise ValueError

    if isinstance(output,numpy.ndarray):
        if isinstance(output.tolist(),(int,float)):
            output=[output]
    return mrr, best_path_index, path_predicted_correct, output


# 1
def answer_question(qa, data, relation_level_words, parameter_dict,goldorpred):
    """one question"""
    log = {}
    log['qid']=data['qid']
    log['language_question_normal']=data['question_normal']
    log['language_abstract_question']=data[goldorpred]['abstract_question']
    log['language_gold_path']=data['gold']['path']
    log['language_predicate_path']=None
    log['question'] = None
    log['true_path'] = None
    log['pred_path'] = None
    metrics = {}
    question, question_dep, question_dep_mask_matrix, candidate_paths,candidate_paths_words, goldpathindex, candidates = dl.construct_paths(data,relation_level_words,qald=True,goldorpred=goldorpred)
    log['question'] = question
    log['question_dep'] = question_dep
    no_positive_path=False
    if goldpathindex==-1:
        no_positive_path=True

    # #############Core chain prediction############
    if no_positive_path: #There is no positive path, maybe we do something intelligent
        log['true_path'] = [-1]
        cps = [n.tolist() for n in candidate_paths]
        paths = cps
        paths_words=[n.tolist() for n in candidate_paths_words]
    else:
        paths = [n.tolist() for n in candidate_paths]
        paths_words = [n.tolist() for n in candidate_paths_words]
        log['true_path'] = paths[goldpathindex]

    ##########Converting paths to numpy array#######
    for i in range(len(paths)):
        paths[i] = np.asarray(paths[i])
        paths_words[i] = np.asarray(paths_words[i])
    paths = np.asarray(paths)
    paths_words = np.asarray(paths_words)
    #############################################

    cc_mrr,best_path_index,cc_acc,output=corechain_prediction(
        question=question, question_dep=question_dep, question_dep_mask_matrix=question_dep_mask_matrix,
        paths=paths, paths_words=paths_words, positive_path_index=goldpathindex, no_positive_path=no_positive_path,
        model=parameter_dict['corechainmodel'], dataset=parameter_dict['dataset'], quesans=qa, candidates=candidates)
    ###########################################

    if best_path_index<0:
        log['language_predicate_path'] =''
        log['pred_path'] = ''
    else:
        log['language_predicate_path'] = candidates[best_path_index]
        log['pred_path'] = paths[best_path_index]
    log['candidate_score']=(zip(candidates,output))
    metrics['core_chain_accuracy_counter'] = cc_acc
    metrics['core_chain_mrr_counter'] = cc_mrr
    metrics['num_paths'] = len(paths)
    return log, metrics

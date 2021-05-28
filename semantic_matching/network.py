import torch
import numpy as np
import qelos as q
# from torch.autograd import Variable
# from torch import nn
import components as com
import tensor_utils as tu


class Model(object):
    """
        Boilerplate class which helps others have some common functionality.
        These are made with some debugging/loading and with corechains in mind
    """
    def prepare_save(self):
        pass

    def load_from(self, location):
        # Pull the data from disk
        print('1##########################################')
        # cuda
        # model_dump = torch.load(location)
        # 0419-ywsun
        model_dump = torch.load(location, map_location=torch.device('cpu'))
        # Load parameters
        for key in self.prepare_save():
            # if key !='vectors':
            key[1].load_state_dict(model_dump[key[0]])

    def get_parameter_sum(self):
        sum = 0
        for model in self.prepare_save():
            model_sum = 0
            for x in list(model[1].parameters()):
                model_sum += np.sum(x.data.cpu().numpy().flatten())
            sum += model_sum
        return sum

    def freeze_layer(self,layer):
        for params in layer.parameters():
            params.requires_grad = False

    def unfreeze_layer(self,layer):
        for params in layer.parameters():
            params.requires_grad = True


class QelosSlotPointerModel(Model):

    """
        Eating Denis's shit
    """
    def __init__(self, _parameter_dict, _device,max_num_relations=2,  _debug=False,single=False,config=None):

        self.debug = _debug
        self.parameter_dict = _parameter_dict
        self.device = _device
        self.config=config
        vectors = self.parameter_dict['vectors']

        if vectors is not None:
            self.embedding_layer = torch.nn.Embedding.from_pretrained(torch.FloatTensor(vectors)).to(self.device)
            self.embedding_layer.weight.requires_grad = False
        else:
            self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        if self.debug:
            print("Init Models")

        self.smcnn=com.SMCNN(config=self.config,embedding_layer=self.embedding_layer,device=self.device)
        self.lin=com.Linear(device=self.device)
        if not single:
            self.encoder_q = com.QelosSlotPtrQuestionEncoder(
                number_of_layer=self.parameter_dict['number_of_layer'],
                bidirectional=self.parameter_dict['bidirectional'],
                embedding_dim=self.parameter_dict['embedding_dim'],
                hidden_dim=self.parameter_dict['hidden_size'],
                vocab_size=self.parameter_dict['vocab_size'],
                max_length=self.parameter_dict['max_length'],
                dropout=self.parameter_dict['dropout'],
                embedding_layer=self.embedding_layer,
                enable_layer_norm=False,
                device=_device,
                residual=True,
                mode = 'LSTM',
                dropout_in=self.parameter_dict['dropout_in'],
                dropout_rec=self.parameter_dict['dropout_rec'],
                debug = self.debug).to(self.device)

        self.encoder_p = com.QelosSlotPtrChainEncoder(
            number_of_layer=self.parameter_dict['number_of_layer'],
            embedding_dim=self.parameter_dict['embedding_dim'],
            bidirectional=self.parameter_dict['bidirectional'],
            hidden_dim=self.parameter_dict['hidden_size'],
            vocab_size=self.parameter_dict['vocab_size'],
            max_length=self.parameter_dict['max_length'],
            dropout=self.parameter_dict['dropout'],
            embedding_layer=self.embedding_layer,
            enable_layer_norm=False,
            device=_device,
            residual=False,
            mode = 'LSTM',
            dropout_in=self.parameter_dict['dropout_in'],
            dropout_rec=self.parameter_dict['dropout_rec'],
            debug = self.debug).to(self.device)


    def train(self, data, optimizer, loss_fn, device):
       return self._train_pairwise_(data, optimizer, loss_fn, device)

    '''original'''
    def _train_pairwise_(self, data, optimizer, loss_fn, device):
        ques_batch, pos_1_batch, pos_2_batch, pos_3_batch, pos_4_batch, neg_1_batch, neg_2_batch, neg_3_batch, neg_4_batch, y_label = \
            data['ques_batch'], data['pos_rel1_batch'], data['pos_rel2_batch'], data['pos_rel3_batch'], data[
                'pos_rel4_batch'], \
            data['neg_rel1_batch'], data['neg_rel2_batch'], data['neg_rel3_batch'], data['neg_rel4_batch'], data[
                'y_label']
        # ques_batch, pos_1_batch, pos_2_batch, neg_1_batch, neg_2_batch, y_label = \
        #     data['ques_batch'], data['pos_rel1_batch'], data['pos_rel2_batch'], \
        #     data['neg_rel1_batch'], data['neg_rel2_batch'], data['y_label']
        optimizer.zero_grad()
        # Have to manually check if the 2nd paths holds anything in this batch.
        # If not, we have to pad everything up with zeros, or even call a limited part of the comparison module.
        pos_2_batch = tu.no_one_left_behind(pos_2_batch)
        pos_3_batch = tu.no_one_left_behind(pos_3_batch)
        pos_4_batch = tu.no_one_left_behind(pos_4_batch)
        neg_2_batch = tu.no_one_left_behind(neg_2_batch)
        neg_3_batch = tu.no_one_left_behind(neg_3_batch)
        neg_4_batch = tu.no_one_left_behind(neg_4_batch)
        # assert torch.mean((torch.sum(pos_2_batch, dim=-1) != 0).float()) == 1
        # assert torch.mean((torch.sum(pos_1_batch, dim=-1) != 0).float()) == 1
        # Encoding all the data
        ques_encoded,_ = self.encoder_q(tu.trim(ques_batch))
        pos_encoded = self.encoder_p(tu.trim(pos_1_batch), tu.trim(pos_2_batch),tu.trim(pos_3_batch), tu.trim(pos_4_batch))
        neg_encoded = self.encoder_p(tu.trim(neg_1_batch), tu.trim(neg_2_batch),tu.trim(neg_3_batch), tu.trim(neg_4_batch))
        # Pass them to the comparison module
        pos_scores = torch.sum(ques_encoded * pos_encoded, dim=-1)
        neg_scores = torch.sum(ques_encoded * neg_encoded, dim=-1)
        '''
            If `y == 1` then it assumed the first input should be ranked higher
            (have a larger value) than the second input, and vice-versa for `y == -1`
        '''
        loss = loss_fn(pos_scores, neg_scores, y_label)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.encoder_q.parameters(), .5)
        # torch.nn.utils.clip_grad_norm_(self.encoder_p.parameters(), .5)
        optimizer.step()
        return loss

    '''original'''
    # def predict(self, question, paths,paths_words, paths_rel1, paths_rel2,paths_rel3,paths_rel4, device,attention_value=False):
    def predict(self, question, question_dep = None, paths = None, paths_words = None,
                paths_rel1=None, paths_rel2=None, paths_rel3=None, paths_rel4=None,
                attention_value=False, device=None):
        # question = question, question_dep = question_dep,
        # paths = paths, paths_words = paths_words,
        # paths_rel1 = paths_rel1, paths_rel2 = paths_rel2, paths_rel3 = paths_rel3, paths_rel4 = paths_rel4, device = device
        """
            Same code works for both pairwise or pointwise
        """
        with torch.no_grad():
            self.encoder_q.eval()
            self.encoder_p.eval()

            # Have to manually check if the 2nd paths holds anything in this batch.
            # If not, we have to pad everything up with zeros, or even call a limited part of the comparison module.
            paths_rel2 = tu.no_one_left_behind(paths_rel2)
            paths_rel3 = tu.no_one_left_behind(paths_rel3)
            paths_rel4 = tu.no_one_left_behind(paths_rel4)

            # Encoding all the data
            ques_encoded,attention_score = self.encoder_q(tu.trim(question))
            # print(paths_rel1)
            path_encoded = self.encoder_p(tu.trim(paths_rel1), tu.trim(paths_rel2),tu.trim(paths_rel3), tu.trim(paths_rel4))

            # Pass them to the comparison module
            score = torch.sum(ques_encoded * path_encoded, dim=-1)

            self.encoder_q.train()
            self.encoder_p.train()
            if attention_value:
                return score,attention_score
            else:
                return score

    def prepare_save1(self):
        return [('encoder_q', self.encoder_q), ('encoder_p', self.encoder_p)]

    def prepare_save(self):
        return [('encoder_q', self.encoder_q), ('encoder_p', self.encoder_p)]
        # return [('smcnn', self.smcnn),('encoder_q', self.encoder_q), ('encoder_p', self.encoder_p),('lin',self.lin)]


# score 0 bert
class Bert_Scorer(Model):

    def __init__(self, _parameter_dict, _device,  _debug=False):

        self.debug = _debug
        self.parameter_dict = _parameter_dict
        self.device = _device
        if self.debug:
            print("Init Models")
        #         new_vectors = self.parameter_dict['vectors']
        #         pretrained_weights['encoder.weight'] = T(new_vectors)
        #         pretrained_weights.pop('encoder_with_dropout.embed.weight')
        #         pretrained_weights['encoder_with_dropout.embed.weight'] = T(np.copy(new_vectors))
        bert = q.bert.TransformerBERT.load_from_dir("data/bert/bert-base")
        self.encoder = com.AdaptedBERTEncoderPair(bert, oldvocab=self.parameter_dict['vocab'], numout=0)
        #         pretrained_weights = torch.load('transformer_encoder.wt', map_location= lambda storage, loc: storage)
        #         self.encoder.load_state_dict(pretrained_weights)
        self.encoder = self.encoder.to(self.device)

    def train(self, data, optimizer, loss_fn, device):
        # Maybe add batch reset
        return self._train_pairwise_(data, optimizer, loss_fn, device)

    def _train_pairwise_(self, data, optimizer, loss_fn, device):
        '''
            Given data, passes it through model, inited in constructor, returns loss and updates the weight
            :params data: {batch of question, pos paths, neg paths and dummy y labels}
            :params optimizer: torch.optim object
            :params loss fn: torch.nn loss object
            :params device: torch.device object
            returns loss
        '''
        self.encoder.train()
        # Unpacking the data and model from args
        ques_batch, pos_batch, pos_batch_words,neg_batch,neg_batch_words, y_label \
            = data['ques_batch'], data['pos_batch'],data['pos_batch_words'], \
                                                    data['neg_batch'],data['neg_batch_words'],\
                                                    data['y_label']
        optimizer.zero_grad()
        ques_batch = tu.trim(ques_batch)
        pos_batch_words = tu.trim(pos_batch_words)
        neg_batch_words = tu.trim(neg_batch_words)
        pos_scores = self.encoder(ques_batch, pos_batch_words)
        neg_scores = self.encoder(ques_batch, neg_batch_words)
        loss = loss_fn(pos_scores, neg_scores, y_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), .5)
        optimizer.step()
        return loss

    def predict(self, question, question_dep=None, paths=None, paths_words=None,
        paths_rel1=None, paths_rel2=None, paths_rel3=None, paths_rel4=None, attention_value = False, device = None):
        """
            Same code works for both pairwise or pointwise
        """
        with torch.no_grad():
            self.encoder.eval()
            ques = tu.trim(question)
            paths = tu.trim(paths_words)
            score = self.encoder(ques, paths)
            self.encoder.train()
            return score

    @property
    def layers(self):
        raise NotImplementedError

    def prepare_save(self):
        return [('encoder', self.encoder)]


# score 1
class Bert_Scorer_slotptr(Model):

    def __init__(self, _parameter_dict, _device,max_num_relations=2, _debug=False):
        self.debug = _debug
        self.parameter_dict = _parameter_dict
        self.device = _device
        if self.debug:
            print("Init Models")
        bert = q.bert.TransformerBERT.load_from_dir('data/bert/bert-base')
        # bert = q.bert.TransformerBERT.load_from_dir("bert")
        self.encoder = com.AdaptedBERTEncoderPairSlotPtr(bert,max_num_relations=max_num_relations, oldvocab=self.parameter_dict['vocab'], numout=0)
        # self.encoder = com.AdaptedBERTEncoderPairSlotPtr_LLZHANG(bert,max_num_relations=max_num_relations, oldvocab=self.parameter_dict['vocab'], numout=0)
        # pretrained_weights = torch.load('transformer_qald/slotptr_init_wt_10_.wt', map_location= lambda storage, loc: storage)
        # self.encoder.load_state_dict(pretrained_weights)
        self.encoder = self.encoder.to(self.device)

    def train(self, data, optimizer, loss_fn, device):
        # Maybe add batch reset
        return self._train_pairwise_(data, optimizer, loss_fn, device)

    def _train_pairwise_(self, data, optimizer, loss_fn, device):
        '''
            Given data, passes it through model, inited in constructor, returns loss and updates the weight
            :params data: {batch of question, pos paths, neg paths and dummy y labels}
            :params optimizer: torch.optim object
            :params loss fn: torch.nn loss object
            :params device: torch.device object
            returns loss
        '''
        self.encoder.train()
        # Unpacking the data and model from args
        ques_batch, \
        pos_1_batch, pos_2_batch, pos_3_batch, pos_4_batch, \
        neg_1_batch, neg_2_batch, neg_3_batch, neg_4_batch, \
        y_label = \
            data['ques_batch'],\
            data['pos_rel1_batch'], data['pos_rel2_batch'],data['pos_rel3_batch'], data['pos_rel4_batch'], \
            data['neg_rel1_batch'], data['neg_rel2_batch'],data['neg_rel3_batch'], data['neg_rel4_batch'],\
            data['y_label']
        optimizer.zero_grad()
        pos_2_batch = tu.no_one_left_behind(pos_2_batch)
        pos_3_batch = tu.no_one_left_behind(pos_3_batch)
        pos_4_batch = tu.no_one_left_behind(pos_4_batch)
        neg_2_batch = tu.no_one_left_behind(neg_2_batch)
        neg_3_batch = tu.no_one_left_behind(neg_3_batch)
        neg_4_batch = tu.no_one_left_behind(neg_4_batch)
        pos_scores = self.encoder(tu.trim(ques_batch), tu.trim(pos_1_batch), tu.trim(pos_2_batch), tu.trim(pos_3_batch), tu.trim(pos_4_batch))
        neg_scores = self.encoder(tu.trim(ques_batch), tu.trim(neg_1_batch), tu.trim(neg_2_batch),tu.trim(neg_3_batch), tu.trim(neg_4_batch))
        loss = loss_fn(pos_scores, neg_scores, y_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), .5)
        optimizer.step()
        return loss

    def predict(self, question, question_dep=None, paths=None, paths_words=None, paths_rel1=None, paths_rel2=None, paths_rel3=None, paths_rel4=None, attention_value=False, device=None):
        """
            Same code works for both pairwise or pointwise
        """
        with torch.no_grad():
            self.encoder.eval()
            paths_rel2 = tu.no_one_left_behind(paths_rel2)
            paths_rel3 = tu.no_one_left_behind(paths_rel3)
            paths_rel4 = tu.no_one_left_behind(paths_rel4)
            score = self.encoder(tu.trim(question), tu.trim(paths_rel1), tu.trim(paths_rel2),tu.trim(paths_rel3),tu.trim(paths_rel4))
            self.encoder.train()
            return score

    @property
    def layers(self):
        raise NotImplementedError

    def prepare_save(self):
        return [('encoder', self.encoder)]


# score 12
class Bert_Scorer_slotptr_w_dep(Model):

    def __init__(self, _parameter_dict, _device,max_num_relations=2, _debug=False):
        self.debug = _debug
        self.parameter_dict = _parameter_dict
        self.device = _device
        if self.debug:
            print("Init Models")
        bert = q.bert.TransformerBERT.load_from_dir('data/bert/bert-base')
        # bert = q.bert.TransformerBERT.load_from_dir("bert")
        """score12"""
        self.encoder = com.AdptedBERTEncoderPairSlotDepPathPtr(bert, max_num_relations=max_num_relations, oldvocab=self.parameter_dict['vocab'], numout=0)
        # pretrained_weights = torch.load('transformer_qald/slotptr_init_wt_10_.wt', map_location= lambda storage, loc: storage)
        # self.encoder.load_state_dict(pretrained_weights)
        self.encoder = self.encoder.to(self.device)

    def train(self, data, optimizer, loss_fn, device):
        # Maybe add batch reset
        return self._train_pairwise_(data, optimizer, loss_fn, device)

    def _train_pairwise_(self, data, optimizer, loss_fn, device):
        '''
            Given data, passes it through model, inited in constructor, returns loss and updates the weight
            :params data: {batch of question, pos paths, neg paths and dummy y labels}
            :params optimizer: torch.optim object
            :params loss fn: torch.nn loss object
            :params device: torch.device object
            returns loss
        '''
        self.encoder.train()
        ques_batch, dep_batch, dep_mask_batch,\
        pos_1_batch, pos_2_batch,pos_3_batch, pos_4_batch, \
        neg_1_batch, neg_2_batch, neg_3_batch, neg_4_batch, y_label = \
            data['ques_batch'], data['ques_dep_batch'], data['ques_dep_mask_batch'], \
            data['pos_rel1_batch'], data['pos_rel2_batch'],data['pos_rel3_batch'], data['pos_rel4_batch'], \
            data['neg_rel1_batch'], data['neg_rel2_batch'],data['neg_rel3_batch'], data['neg_rel4_batch'], \
            data['y_label']
        optimizer.zero_grad()
        pos_2_batch = tu.no_one_left_behind(pos_2_batch)
        pos_3_batch = tu.no_one_left_behind(pos_3_batch)
        pos_4_batch = tu.no_one_left_behind(pos_4_batch)
        neg_2_batch = tu.no_one_left_behind(neg_2_batch)
        neg_3_batch = tu.no_one_left_behind(neg_3_batch)
        neg_4_batch = tu.no_one_left_behind(neg_4_batch)
        pos_scores = self.encoder(tu.trim(ques_batch), dep_batch, dep_mask_batch,
                                  tu.trim(pos_1_batch), tu.trim(pos_2_batch), tu.trim(pos_3_batch), tu.trim(pos_4_batch))
        neg_scores = self.encoder(tu.trim(ques_batch), dep_batch, dep_mask_batch,
                                  tu.trim(neg_1_batch), tu.trim(neg_2_batch), tu.trim(neg_3_batch), tu.trim(neg_4_batch))
        loss = loss_fn(pos_scores, neg_scores, y_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), .5)
        # torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), .5, norm_type=2)
        #梯度裁剪原理：既然在BP过程中会产生梯度消失.
        # 那么最简单粗暴的方法，设定阈值，当梯度小于阈值时，更新的梯度为阈值
        optimizer.step()
        # optimizer.zero_grad()
        return loss

    def predict(self, question=None, question_dep=None, question_dep_mask=None, paths=None, paths_words=None, paths_rel1=None, paths_rel2=None, paths_rel3=None, paths_rel4=None, attention_value=False, device=None):
        """
            Same code works for both pairwise or pointwise
        """
        with torch.no_grad():
            self.encoder.eval()
            paths_rel2 = tu.no_one_left_behind(paths_rel2)
            paths_rel3 = tu.no_one_left_behind(paths_rel3)
            paths_rel4 = tu.no_one_left_behind(paths_rel4)
            score = self.encoder(tu.trim(question), question_dep, question_dep_mask, tu.trim(paths_rel1), tu.trim(paths_rel2),tu.trim(paths_rel3),tu.trim(paths_rel4))
            # score = score.squeeze() # v0.2 in order to solve: mrr = mrr_output.index(positive_path_index) + 1.0
            self.encoder.train()
            return score

    @property
    def layers(self):
        raise NotImplementedError

    def prepare_save(self):
        return [('encoder', self.encoder)]


# score 2
class Bert_DepMatch(Model):
    def __init__(self, _parameter_dict, _device,max_num_relations=2, _debug=False):
        self.debug = _debug
        self.parameter_dict = _parameter_dict
        self.device = _device
        if self.debug:
            print("Init Models")
        bert = q.bert.TransformerBERT.load_from_dir('data/bert/bert-base')
        # bert = q.bert.TransformerBERT.load_from_dir("bert")
        self.encoder = com.AdptedBERTEncoderDepPath(bert, max_num_relations=max_num_relations, oldvocab=self.parameter_dict['vocab'], numout=0)
        # pretrained_weights = torch.load('transformer_qald/slotptr_init_wt_10_.wt', map_location= lambda storage, loc: storage)
        # self.encoder.load_state_dict(pretrained_weights)
        self.encoder = self.encoder.to(self.device)

    def train(self, data, optimizer, loss_fn, device):
        return self._train_pairwise_(data, optimizer, loss_fn, device)

    def _train_pairwise_(self, data, optimizer, loss_fn, device):
        '''
            Given data, passes it through model, inited in constructor, returns loss and updates the weight
            :params data: {batch of question, pos paths, neg paths and dummy y labels}
            :params optimizer: torch.optim object
            :params loss fn: torch.nn loss object
            :params device: torch.device object
            returns loss
        '''
        self.encoder.train()
        assert 'ques_dep_batch' in data
        dep_batch, dep_mask_batch,\
        pos_1_batch, pos_2_batch, pos_3_batch, pos_4_batch, \
        neg_1_batch, neg_2_batch, neg_3_batch, neg_4_batch, y_label = data['ques_dep_batch'], data['ques_dep_mask_batch'],\
            data['pos_rel1_batch'], data['pos_rel2_batch'],data['pos_rel3_batch'], data['pos_rel4_batch'], \
            data['neg_rel1_batch'], data['neg_rel2_batch'],data['neg_rel3_batch'], data['neg_rel4_batch'], \
            data['y_label']
        optimizer.zero_grad()
        pos_2_batch = tu.no_one_left_behind(pos_2_batch)
        pos_3_batch = tu.no_one_left_behind(pos_3_batch)
        pos_4_batch = tu.no_one_left_behind(pos_4_batch)
        neg_2_batch = tu.no_one_left_behind(neg_2_batch)
        neg_3_batch = tu.no_one_left_behind(neg_3_batch)
        neg_4_batch = tu.no_one_left_behind(neg_4_batch)
        dep_batch = tu.trim(dep_batch)
        pos_scores = self.encoder(dep_batch, dep_mask_batch, tu.trim(pos_1_batch), tu.trim(pos_2_batch), tu.trim(pos_3_batch), tu.trim(pos_4_batch))
        neg_scores = self.encoder(tu.trim(dep_batch), dep_mask_batch, tu.trim(neg_1_batch), tu.trim(neg_2_batch), tu.trim(neg_3_batch), tu.trim(neg_4_batch))
        loss = loss_fn(pos_scores, neg_scores, y_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), .5)
        # torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), .5, norm_type=2)
        #梯度裁剪原理：既然在BP过程中会产生梯度消失.
        # 那么最简单粗暴的方法，设定阈值，当梯度小于阈值时，更新的梯度为阈值
        optimizer.step()
        return loss

    def predict(self, question=None, question_dep=None, question_dep_mask=None, paths=None, paths_words=None, paths_rel1=None, paths_rel2=None, paths_rel3=None, paths_rel4=None, attention_value=False, device=None):
        """
            Same code works for both pairwise or pointwise
        """
        with torch.no_grad():
            assert question_dep is not None
            self.encoder.eval()
            paths_rel2 = tu.no_one_left_behind(paths_rel2)
            paths_rel3 = tu.no_one_left_behind(paths_rel3)
            paths_rel4 = tu.no_one_left_behind(paths_rel4)
            score = self.encoder(tu.trim(question_dep), question_dep_mask, tu.trim(paths_rel1), tu.trim(paths_rel2),tu.trim(paths_rel3),tu.trim(paths_rel4))
            # score = score.squeeze() # v0.2 in order to solve: mrr = mrr_output.index(positive_path_index) + 1.0
            self.encoder.train()
            return score

    @property
    def layers(self):
        raise NotImplementedError

    def prepare_save(self):
        return [('encoder', self.encoder)]

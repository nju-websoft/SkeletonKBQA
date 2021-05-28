import torch
from fix_seed import set_seed
set_seed(1)
print('#randam:\t', torch.rand(5))

import argparse
import json
import pickle
import numpy as np
import time
import embeddings_interface
import onefile_valid as one_valid
import auxiliary as aux
from configs import config_loader as cl
import data_loader as dl
from torch.utils.data import DataLoader
import network as net
from configs.config_loader import Config


def load_data(data,parameter_dict,schema='default',shuffle = False):
    td = dl.TrainingDataGenerator(data, parameter_dict['max_length'], parameter_dict['_neg_paths_per_epoch_train'], parameter_dict['batch_size'], parameter_dict['total_negative_samples'],schema=schema)
    return DataLoader(td, shuffle=shuffle)


def curatail_padding(data,parameter_dict):
    '''
        Since schema is already implicitly defined/present in the parameter_dict['rel1_pad']
    '''
    data['valid_neg_paths'] = data['valid_neg_paths'][:, :, :parameter_dict['rel_pad']]
    data['valid_neg_paths_words'] = data['valid_neg_paths_words'][:, :, :parameter_dict['rel_pad']]
    data['valid_pos_paths'] = data['valid_pos_paths'][:, :parameter_dict['rel_pad']]
    data['valid_pos_paths_words'] = data['valid_pos_paths_words'][:, :parameter_dict['rel_pad']]
    if parameter_dict['schema'] == 'slotptr':
        data['valid_neg_paths_rel1_sp'] = data['valid_neg_paths_rel1_sp'][:, :, :parameter_dict['rel1_pad']]
        data['valid_neg_paths_rel2_sp'] = data['valid_neg_paths_rel2_sp'][:, :, :parameter_dict['rel1_pad']]
        data['valid_neg_paths_rel3_sp'] = data['valid_neg_paths_rel3_sp'][:, :, :parameter_dict['rel1_pad']]
        data['valid_neg_paths_rel4_sp'] = data['valid_neg_paths_rel4_sp'][:, :, :parameter_dict['rel1_pad']]
        data['valid_pos_paths_rel1_sp'] = data['valid_pos_paths_rel1_sp'][:, :parameter_dict['rel1_pad']]
        data['valid_pos_paths_rel2_sp'] = data['valid_pos_paths_rel2_sp'][:, :parameter_dict['rel1_pad']]
        data['valid_pos_paths_rel3_sp'] = data['valid_pos_paths_rel3_sp'][:, :parameter_dict['rel1_pad']]
        data['valid_pos_paths_rel4_sp'] = data['valid_pos_paths_rel4_sp'][:, :parameter_dict['rel1_pad']]
    return data


"""3"""


def valid_or_test(device='cuda', _dataset='lcquad', training_model='bert_slotptr_w_dep', corechain_model=None, model_save_location=None, mode='valid'):
    '''
    :param device: # cuda 'cpu'
    :return:
    '''
    print('#:', mode, _dataset, training_model)
    _dataset_specific_data_dir = 'data/data/%(dataset)s/' % {'dataset': _dataset}
    rel_level_words = aux.read_json(_dataset_specific_data_dir + 'relation_level_words')
    parameter_dict = cl.corechain_parameters(dataset=_dataset, training_model=training_model, config_file='configs/macros.cfg')
    parameter_dict['_dataset_specific_data_dir'] = _dataset_specific_data_dir
    parameter_dict['_model_dir'] = 'data/models/'
    parameter_dict['corechainmodel'] = training_model
    parameter_dict['dataset'] = _dataset

    _dataset_specific_data_dir, _model_specific_data_dir, _file, _validatation_file, \
    _max_sequence_length, _neg_paths_per_epoch_train, \
    _neg_paths_per_epoch_validation, _training_split, _validation_split = aux.data_loading_parameters(_dataset,parameter_dict,runtime=True)

    assert mode in ['valid', 'test']
    if mode == 'valid':
        goldorpred = 'gold'
        _data, _vectors = dl.create_dataset_runtime(file=_validatation_file, _dataset_specific_data_dir=_dataset_specific_data_dir)
    elif mode == 'test':
        goldorpred = 'pred'
        _data, _vectors = dl.create_dataset_runtime(file=_file, _dataset_specific_data_dir=_dataset_specific_data_dir)
    parameter_dict['vectors'] = _vectors
    parameter_dict['vocab'] = pickle.load(open('resources/vocab_gl.pickle', 'rb'))
    Logging = {}
    Logging['runtime'] = []

    if _dataset in ['lcquad']: max_num_relations = 2
    elif _dataset in ["graphq"]: max_num_relations = 3
    elif _dataset in ['complexwebq', 'cwq']: max_num_relations = 4
    else: max_num_relations = 2

    quesans = one_valid.QuestionAnswering(parameters=parameter_dict, device=device, _dataset=_dataset, max_num_relations=max_num_relations, debug=False,
                                          corechain_model=corechain_model, model_save_location=model_save_location)
    core_chain_acc_log = []
    core_chain_mrr_log = []
    startindex = 0
    error_list = []
    for index, data in enumerate(_data[startindex:]):
        try:
            log, metrics = one_valid.answer_question(qa=quesans, data=data, relation_level_words=rel_level_words, parameter_dict=parameter_dict, goldorpred=goldorpred)
        except Exception as e:
            print('error')
            error_list.append(data['qid'])
            continue
        Logging['runtime'].append({'log': log, 'metrics': metrics})
        core_chain_acc_log.append(metrics['core_chain_accuracy_counter'])
        core_chain_mrr_log.append(metrics['core_chain_mrr_counter'])
        if index % 20 == 0: print("#%s" % index, "\t\bAcc: ", np.mean(core_chain_acc_log))
    print('#error list:\t', error_list)
    Logging['num'] = len(core_chain_acc_log)
    Logging['mean_core_chain_accuracy_counter'] = np.mean(core_chain_acc_log)
    Logging['mean_core_chain_mrr_counter'] = np.mean(core_chain_mrr_log)
    if mode == 'test':
        pickle.dump(Logging, open(model_save_location + '/test_result.pickle', 'wb'), protocol=4)
    elif mode == 'valid':
        pickle.dump(Logging, open(model_save_location + '/valid_result.pickle', 'wb'), protocol=4)
    return np.mean(core_chain_acc_log)


"""2"""


def training_loop(training_model, parameter_dict,modeler,train_loader, optimizer,loss_func, data, dataset, device, problem='core_chain', curtail_padding_rel=True):
    model_save_location = aux.save_location(problem, training_model, dataset, parameter_dict)
    aux_save_information = {'epoch': 0, 'test_accuracy': 0.0, 'validation_accuracy': 0.0, 'parameter_dict': parameter_dict }
    train_loss = []
    valid_accuracy = []
    best_validation_accuracy = 0
    if parameter_dict['schema'] == 'slotptr': parameter_dict['rel1_pad'] = parameter_dict['relsp_pad']
    print("the dataset is ", dataset)
    if curtail_padding_rel:
        data = curatail_padding(data, parameter_dict)
    try:
        iters_not_improved = 0
        early_stop = False
        patience = 5
        for epoch in range(parameter_dict['epochs']):
            print("Epoch: ", epoch, "/", parameter_dict['epochs'])
            if early_stop:
                print("Early stopping. Epoch: {}, Best Dev. Acc: {}".format(epoch, best_validation_accuracy))
                break
            epoch_loss = []
            epoch_time = time.time()
            print(len(train_loader))
            for i_batch, sample_batched in enumerate(train_loader):
                batch_time = time.time()
                if parameter_dict['schema'] != 'default':
                    ques_batch = torch.tensor(np.reshape(sample_batched[0][0], (-1, parameter_dict['max_length'])),dtype=torch.long, device=device)
                    ques_dep_batch = torch.tensor(np.reshape(sample_batched[0][1], (-1, sample_batched[0][1].shape[-2], sample_batched[0][1].shape[-1])), dtype=torch.long, device=device)
                    ques_dep_mask_batch = torch.tensor(np.reshape(sample_batched[0][2], (-1, sample_batched[0][2].shape[-1])), dtype=torch.float32, device=device)

                    pos_batch = torch.tensor(np.reshape(sample_batched[0][3], (-1, parameter_dict['max_length'])), dtype=torch.long, device=device)
                    pos_batch_words = torch.tensor(np.reshape(sample_batched[0][13 if parameter_dict['schema'] != 'default' else 5], (-1, parameter_dict['max_length'])), dtype=torch.long, device=device)

                    neg_batch = torch.tensor(np.reshape(sample_batched[0][8], (-1, parameter_dict['max_length'])), dtype=torch.long, device=device)
                    neg_batch_words = torch.tensor(np.reshape(sample_batched[0][14 if parameter_dict['schema'] != 'default' else 4],(-1, parameter_dict['max_length'])), dtype=torch.long, device=device)
                    data['dummy_y'] = torch.ones(ques_batch.shape[0], device=device)

                    pos_rel1_batch = torch.tensor(np.reshape(sample_batched[0][4], (-1, parameter_dict['max_length'])), dtype=torch.long, device=device)
                    pos_rel2_batch = torch.tensor(np.reshape(sample_batched[0][5], (-1, parameter_dict['max_length'])), dtype=torch.long, device=device)
                    pos_rel3_batch = torch.tensor(np.reshape(sample_batched[0][6], (-1, parameter_dict['max_length'])), dtype=torch.long, device=device)
                    pos_rel4_batch = torch.tensor(np.reshape(sample_batched[0][7], (-1, parameter_dict['max_length'])), dtype=torch.long, device=device)
                    neg_rel1_batch = torch.tensor(np.reshape(sample_batched[0][9], (-1, parameter_dict['max_length'])), dtype=torch.long, device=device)
                    neg_rel2_batch = torch.tensor(np.reshape(sample_batched[0][10], (-1, parameter_dict['max_length'])), dtype=torch.long, device=device)
                    neg_rel3_batch = torch.tensor(np.reshape(sample_batched[0][11], (-1, parameter_dict['max_length'])), dtype=torch.long, device=device)
                    neg_rel4_batch = torch.tensor(np.reshape(sample_batched[0][12], (-1, parameter_dict['max_length'])),dtype=torch.long, device=device)

                    data_batch = {
                        'ques_batch': ques_batch,
                        'ques_dep_batch': ques_dep_batch,
                        'ques_dep_mask_batch': ques_dep_mask_batch,
                        'pos_batch': pos_batch[:, :parameter_dict['rel_pad']],
                        'pos_batch_words': pos_batch_words[:, :parameter_dict['rel_pad']],
                        'neg_batch': neg_batch[:, :parameter_dict['rel_pad']],
                        'neg_batch_words': neg_batch_words[:, :parameter_dict['rel_pad']],
                        'y_label': data['dummy_y'],
                        'pos_rel1_batch': pos_rel1_batch[:, :parameter_dict['rel1_pad']],
                        'pos_rel2_batch': pos_rel2_batch[:, :parameter_dict['rel1_pad']],
                        'pos_rel3_batch': pos_rel3_batch[:, :parameter_dict['rel1_pad']],
                        'pos_rel4_batch': pos_rel4_batch[:, :parameter_dict['rel1_pad']],
                        'neg_rel1_batch': neg_rel1_batch[:, :parameter_dict['rel1_pad']],
                        'neg_rel2_batch': neg_rel2_batch[:, :parameter_dict['rel1_pad']],
                        'neg_rel3_batch': neg_rel3_batch[:, :parameter_dict['rel1_pad']],
                        'neg_rel4_batch': neg_rel4_batch[:, :parameter_dict['rel1_pad']],
                    }
                else:
                    ques_batch = torch.tensor(np.reshape(sample_batched[0][0],(-1, parameter_dict['max_length'])),dtype=torch.long, device=device)
                    pos_batch = torch.tensor(np.reshape(sample_batched[0][3],(-1, parameter_dict['max_length'])), dtype=torch.long, device=device)
                    pos_batch_words = torch.tensor(np.reshape(sample_batched[0][5],(-1, parameter_dict['max_length'])), dtype=torch.long, device=device)
                    neg_batch = torch.tensor(np.reshape(sample_batched[0][4],(-1, parameter_dict['max_length'])), dtype=torch.long, device=device)
                    neg_batch_words = torch.tensor(np.reshape(sample_batched[0][6],(-1, parameter_dict['max_length'])), dtype=torch.long, device=device)
                    data['dummy_y'] = torch.ones(ques_batch.shape[0], device=device)
                    data_batch = {
                        'ques_batch': ques_batch,
                        'pos_batch': pos_batch[:, :parameter_dict['rel_pad']],
                        'pos_batch_words': pos_batch_words[:, :parameter_dict['rel_pad']],
                        'neg_batch': neg_batch[:, :parameter_dict['rel_pad']],
                        'neg_batch_words': neg_batch_words[:, :parameter_dict['rel_pad']],
                        'y_label': data['dummy_y']}
                loss = modeler.train(data=data_batch, optimizer=optimizer, loss_fn=loss_func, device=device)
                epoch_loss.append(loss.item())
                if i_batch % 50 == 0:
                    print("Batch:\t%d" % i_batch, "/%d\t: " % (parameter_dict['batch_size']), "%s" % (time.time() - batch_time), "\t%s" % (time.time() - epoch_time), "\t%s" % (str(loss.item())),
                          end=None if i_batch + 1 == int(int(i_batch) / parameter_dict['batch_size']) else "\n")
            # EPOCH LEVEL  Track training loss
            train_loss.append(epoch_loss)
            # Run on validation set
            valid_acc = valid_or_test(device=device.type, _dataset=dataset,training_model=training_model,
                                      corechain_model=modeler, model_save_location=model_save_location, mode='valid')
            valid_accuracy.append(valid_acc)
            if valid_accuracy[-1] > best_validation_accuracy:
                best_validation_accuracy = valid_accuracy[-1]
                aux_save_information['epoch'] = epoch
                aux_save_information['validation_accuracy'] = best_validation_accuracy
                aux.save_model(model_save_location, modeler, model_name='model.torch', epochs=epoch, optimizer=optimizer, accuracy=best_validation_accuracy, aux_save_information=aux_save_information)
                iters_not_improved = 0
            else:
                iters_not_improved += 1
                if iters_not_improved >= patience:
                    early_stop = True
            # Resample new negative paths per epoch and shuffle all data
            train_loader.dataset.shuffle()
            print("Time: %s\t" % (time.time() - epoch_time), "Loss: %s\t" % (sum(epoch_loss)), "Valdacc: %s\t" % (valid_accuracy[-1]), "BestValidAcc: %s\n" % (best_validation_accuracy))
        return train_loss, valid_accuracy, model_save_location
    except KeyboardInterrupt:
        print('-' * 89)
        return train_loss, valid_accuracy, model_save_location


"""1"""


def train_interface():
    """args"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', action='store', dest='dataset', help='dataset includes lcquad, graphq, webqsp, cwq', default='cwq')
    parser.add_argument('-model', action='store', dest='model', help='name of the model to use', default='bert_slotptr_w_dep') #bert_slotptr, bert_slotptr_w_dep, bert_deppath
    parser.add_argument('-device', action='store', dest='device', help='cuda for gpu else cpu', default='cpu') #cpu cuda
    parser.add_argument("-learning_rate", default=1e-5, type=float, help="Total number of training epoches to perform")
    parser.add_argument("-batch_size", default=16, type=int, help="The epoches of training") #1 20, 30
    parser.add_argument("-struct", default='skeleton', type=str, help="The epoches of training") #1 20, 30
    args = parser.parse_args()

    """parameter_dict"""
    device = torch.device(args.device)
    training_model = args.model
    _dataset = args.dataset
    parameter_dict = cl.corechain_parameters(dataset=_dataset, training_model=training_model, config_file='configs/macros.cfg')
    parameter_dict['batch_size'] = args.batch_size
    parameter_dict['learning_rate'] = args.learning_rate
    parameter_dict['struct'] = args.struct

    _dataset_specific_data_dir = 'data/data/%(dataset)s/' % {'dataset': _dataset}
    rel_level_words=aux.read_json(_dataset_specific_data_dir+'relation_level_words')
    data = aux.load_data(_dataset=_dataset,_parameter_dict=parameter_dict, _relation_level_words=rel_level_words, _device=device)
    if training_model in ['bert_slotptr', 'slotptr', 'bert_slotptr_w_dep', 'bert_deppath']:
        schema = 'slotptr'
    elif training_model in ['bert']:
        schema = 'default'
    train_loader = load_data(data, parameter_dict, schema=schema)
    parameter_dict['vectors'] = embeddings_interface.vectors
    parameter_dict['schema'] = schema
    parameter_dict['vocab'] = pickle.load(open('resources/vocab_gl.pickle', 'rb'))
    if _dataset in ['lcquad']: max_num_relations=2
    elif _dataset in ['graphq']: max_num_relations=3
    elif _dataset in ['compwebq', 'cwq']: max_num_relations=4
    else: max_num_relations=2
    learning_rate = parameter_dict['learning_rate']

    """modeler"""
    if training_model == 'slotptr':
        config=Config()
        modeler = net.QelosSlotPointerModel(_parameter_dict=parameter_dict,max_num_relations=max_num_relations, _device=device, _debug=False,config=config)
        optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder_q.parameters()))+list(filter(lambda p: p.requires_grad, modeler.encoder_p.parameters())), weight_decay=0.0001, lr=learning_rate) #0.001
    elif training_model == 'bert_slotptr':
        modeler = net.Bert_Scorer_slotptr(_parameter_dict=parameter_dict,_device=device, max_num_relations=max_num_relations, _debug=False)
        optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder.parameters())), lr=learning_rate) #0.00001
    elif training_model == 'bert_slotptr_w_dep':
        modeler = net.Bert_Scorer_slotptr_w_dep(_parameter_dict=parameter_dict, _device=device, max_num_relations=max_num_relations, _debug=False)
        optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder.parameters())), lr=learning_rate) #0.00001
    elif training_model == 'bert_deppath':
        modeler = net.Bert_DepMatch(_parameter_dict=parameter_dict, _device=device, max_num_relations=max_num_relations, _debug=False)
        optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder.parameters())), lr=learning_rate) #0.00001
    elif training_model == 'bert':
        modeler = net.Bert_Scorer(_parameter_dict=parameter_dict, _device=device, _debug=False)
        optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, modeler.encoder.parameters())), lr=learning_rate) #0.00001
    """train"""
    loss_func = torch.nn.MarginRankingLoss(margin=1.0, size_average=False)#0, 0.1, 0.2, 0.5
    train_loss, valid_accuracy, model_save_location = training_loop(training_model=training_model, parameter_dict=parameter_dict, modeler=modeler,
        train_loader=train_loader, optimizer=optimizer, loss_func=loss_func, data=data, dataset=parameter_dict['dataset'], device=device, problem='core_chain')
    print("validation accuracy is , ", max(valid_accuracy))
    print("model saved at, ", model_save_location)
    msl = model_save_location.split('/')
    print(f"model save locaton info {msl}")
    json.dump(train_loss, open(f"{model_save_location}/loss.json", 'w+'))


if __name__ == '__main__':

    train_interface()


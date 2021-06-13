import json
import pickle
import os
import torch
import data_loader as dl


def read_json(pathfile):
    with open(pathfile, 'r', encoding="utf-8") as f:
        data = json.load(f)
    f.close()
    return data


# 2
def data_loading_parameters(dataset,parameter_dict,runtime=False):

    if dataset == 'lcquad':
        _dataset_specific_data_dir = 'data/data/lcquad/'
        _model_specific_data_dir = 'data/data/core_chain/lcquad/'
        # SP_2
        # _file = 'SP_2_E2E_train_lcquad_candidate_path_0201_2014.json'
        # _validatation_file = 'SP_2_E2E_vaild_lcquad_candidate_path_0201_281.json'

        # SP_4
        # _file = 'SP_4_E2E_train_lcquad_candidate_path_0201_1805.json'
        # _validatation_file = 'SP_4_E2E_valid_lcquad_candidate_path_0201_243.json'

        # IR_5
        _file = 'IR_5_E2E_train_skeleton_lcquad_candidate_path_0201_2248.json'
        _validatation_file = 'IR_5_E2E_valid_skeleton_lcquad_candidate_path_0201_317.json'

        # IR_6
        # _file = 'IR_6_E2E_train_dep_lcquad_candidate_path_0201_2248.json'
        # _validatation_file = 'IR_6_E2E_valid_dep_lcquad_candidate_path_0201_317.json'

        # IR_9

        _max_sequence_length = parameter_dict['max_length']
        _neg_paths_per_epoch_train = parameter_dict['_neg_paths_per_epoch_train']
        _neg_paths_per_epoch_validation = parameter_dict['_neg_paths_per_epoch_validation']
        _training_split = .875
        _validation_split = 1.0
        if runtime:
            # SP_2
            # _file = 'SP_2_E2E_test_lcquad_candidate_path_0201_567.json'
            # _validatation_file = 'SP_2_E2E_vaild_lcquad_candidate_path_0201_281.json'

            # SP_4
            # _file = 'SP_4_E2E_test_lcquad_candidate_path_0201_499.json'
            # _validatation_file = 'SP_4_E2E_valid_lcquad_candidate_path_0201_243.json'

            # IR_5
            # _file = 'IR_5_E2E_test_skeleton_lcquad_candidate_path_0201_636.json'
            _file = 'IR_5_E2E_test_skeleton_lcquad_candidate_path_0321_sample.json'
            _validatation_file = 'IR_5_E2E_valid_skeleton_lcquad_candidate_path_0201_317.json'

            # IR_6
            # _file = 'IR_6_E2E_test_dep_lcquad_candidate_path_0201_636.json'
            # _validatation_file = 'IR_6_E2E_valid_dep_lcquad_candidate_path_0201_317.json'

            # IR_9

            _training_split = .0
            _validation_split = 0.0

    elif dataset=='graphq':
        _dataset_specific_data_dir = 'data/data/graphq/'
        _model_specific_data_dir = 'data/data/core_chain/graphq/'
        # SP_2
        # _file = 'SP_2_E2E_train_graphq_candidate_path_0131_1474.json'
        # _validatation_file = 'SP_2_E2E_valid_graphq_candidate_path_0131_159.json'
        # _file = 'SP_2_E2E_train_graphq_candidate_path_0220_1474_v0.2.json'
        # _validatation_file = 'SP_2_E2E_valid_graphq_candidate_path_0220_159_v0.2.json'

        # SP_4
        # _file = 'SP_4_E2E_train_graphq_candidate_path_0131_1403.json'
        # _validatation_file = 'SP_4_E2E_valid_graphq_candidate_path_0131_159.json'
        # _file = 'SP_4_E2E_train_graphq_candidate_path_0220_1403_v0.2.json'
        # _validatation_file = 'SP_4_E2E_valid_graphq_candidate_path_0220_159_v0.2.json'

        # IR_5
        # _file = 'IR_5_E2E_skeleton_train_graphq_candidate_path_0131_1462.json'
        # _validatation_file = 'IR_5_E2E_skeleton_valid_graphq_candidate_path_0131_183.json'
        _file = 'IR_5_v0.1_E2E_skeleton_train_graphq_candidate_path_0225_1510.json'
        _validatation_file = 'IR_5_v0.1_E2E_skeleton_valid_graphq_candidate_path_0225_185.json'

        # IR_6
        # _file = 'IR_6_E2E_dep_train_graphq_candidate_path_0131_1462.json'
        # _validatation_file = 'IR_6_E2E_dep_valid_graphq_candidate_path_0131_183.json'
        # _file = 'IR_6_v0.1_E2E_dep_train_graphq_candidate_path_0225_1510.json'
        # _validatation_file = 'IR_6_v0.1_E2E_dep_valid_graphq_candidate_path_0225_185.json'

        # IR_9

        _max_sequence_length = parameter_dict['max_length']
        _neg_paths_per_epoch_train = parameter_dict['_neg_paths_per_epoch_train']
        _neg_paths_per_epoch_validation = parameter_dict['_neg_paths_per_epoch_validation']
        _training_split = .875
        _validation_split = 1.0
        if runtime:
            # SP_2
            # _file = 'SP_2_E2E_test_graphq_candidate_path_0131_980.json'
            # _validatation_file = 'SP_2_E2E_valid_graphq_candidate_path_0131_159.json'
            # _file = 'SP_2_E2E_test_graphq_candidate_path_0220_1013_v0.2.json'
            # _validatation_file = 'SP_2_E2E_valid_graphq_candidate_path_0220_159_v0.2.json'

            # SP_4
            # _file = 'SP_4_E2E_test_graphq_candidate_path_0131_971.json'
            # _validatation_file = 'SP_4_E2E_valid_graphq_candidate_path_0131_159.json'
            # _file = 'SP_4_E2E_test_graphq_candidate_path_0220_1004_v0.2.json'
            # _validatation_file = 'SP_4_E2E_valid_graphq_candidate_path_0220_159_v0.2.json'

            # IR_5
            # _file = 'IR_5_E2E_skeleton_test_graphq_candidate_path_0131_1100.json'
            # _validatation_file = 'IR_5_E2E_skeleton_valid_graphq_candidate_path_0131_183.json'
            _file = 'IR_5_v0.1_E2E_skeleton_test_graphq_candidate_path_0225_1070.json'
            _validatation_file = 'IR_5_v0.1_E2E_skeleton_valid_graphq_candidate_path_0225_185.json'

            # IR_6
            # _file = 'IR_6_E2E_dep_test_graphq_candidate_path_0131_1070.json'
            # _validatation_file = 'IR_6_E2E_dep_valid_graphq_candidate_path_0131_183.json'
            # _file = 'IR_6_v0.1_E2E_dep_test_graphq_candidate_path_0225_1070.json'
            # _validatation_file = 'IR_6_v0.1_E2E_dep_valid_graphq_candidate_path_0225_185.json'

            # IR_9

            _training_split = .0
            _validation_split = 0.0

    elif dataset=='cwq':
        _dataset_specific_data_dir = 'data/data/cwq/'
        _model_specific_data_dir = 'data/data/core_chain/cwq/'
        # SP_2
        # _file = 'SP_2_E2E_train_cwq_candidate_path_0209_9874.json'
        # _validatation_file = 'SP_2_E2E_dev_cwq_candidate_path_0209_1238.json'
        # _file = 'SP_2_E2E_train_cwq_candidate_path_0217_10861_v0.2.json'
        # _validatation_file = 'SP_2_E2E_dev_cwq_candidate_path_0217_1393_v0.2.json'

        # SP_4
        # _file = 'SP_4_E2E_train_cwq_candidate_path_0210_9171.json'
        # _validatation_file = 'SP_4_E2E_dev_cwq_candidate_path_0210_1138.json'
        # _file = 'SP_4_E2E_train_cwq_candidate_path_0218_10217_v0.2.json'
        # _validatation_file = 'SP_4_E2E_dev_cwq_candidate_path_0218_1303_v0.2.json'

        # IR_5
        # _file = 'IR_5_E2E_train_cwq_candidate_path_0211_12510.json'
        # _validatation_file = 'IR_5_E2E_dev_cwq_candidate_path_0211_1614.json'
        # _file = 'IR_5_v0.1_E2E_train_cwq_candidate_path_0227_11553.json'
        # _validatation_file = 'IR_5_v0.1_E2E_dev_cwq_candidate_path_0227_1465.json'

        # IR_6
        # _file = 'IR_6_E2E_train_cwq_candidate_path_0211_12510.json'
        # _validatation_file = 'IR_6_E2E_dev_cwq_candidate_path_0211_1615.json'
        _file = 'IR_6_v0.1_E2E_train_cwq_candidate_path_0227_11553.json'
        _validatation_file = 'IR_6_v0.1_E2E_dev_cwq_candidate_path_0227_1465.json'

        # IR_9

        _max_sequence_length = parameter_dict['max_length']
        _neg_paths_per_epoch_train = parameter_dict['_neg_paths_per_epoch_train']
        _neg_paths_per_epoch_validation = parameter_dict['_neg_paths_per_epoch_validation']
        _training_split = .875
        _validation_split = 1.0
        if runtime:
            # SP_2
            # _file = 'SP_2_E2E_test_cwq_candidate_path_0209_1256.json'
            # _validatation_file = 'SP_2_E2E_dev_cwq_candidate_path_0209_1238.json'
            # _file = 'SP_2_E2E_test_cwq_candidate_path_0217_1422_v0.2.json'
            # _validatation_file = 'SP_2_E2E_dev_cwq_candidate_path_0217_1393_v0.2.json'

            # SP_4
            # _file = 'SP_4_E2E_test_cwq_candidate_path_0210_1160.json'
            # _validatation_file = 'SP_4_E2E_dev_cwq_candidate_path_0210_1138.json'
            # _file = 'SP_4_E2E_test_cwq_candidate_path_0218_1336_v0.2.json'
            # _validatation_file = 'SP_4_E2E_dev_cwq_candidate_path_0218_1303_v0.2.json'

            # IR_5
            # _file = 'IR_5_E2E_test_cwq_candidate_path_0218_7_temp.json'
            # _validatation_file = 'IR_5_E2E_dev_cwq_candidate_path_0211_1614.json'
            # _file = 'IR_5_v0.1_E2E_test_cwq_candidate_path_0227_1444.json'
            # _validatation_file = 'IR_5_v0.1_E2E_dev_cwq_candidate_path_0227_1465.json'

            # IR_6
            # _file = 'IR_6_E2E_test_cwq_candidate_path_0211_1604.json'
            # _validatation_file = 'IR_6_E2E_dev_cwq_candidate_path_0211_1615.json'
            _file = 'IR_6_v0.1_E2E_test_cwq_candidate_path_0227_1444.json'
            _validatation_file = 'IR_6_v0.1_E2E_dev_cwq_candidate_path_0227_1465.json'

            # IR_9

            _training_split = .0
            _validation_split = 0.0

    return _dataset_specific_data_dir,_model_specific_data_dir,_file,_validatation_file, \
           _max_sequence_length,_neg_paths_per_epoch_train,_neg_paths_per_epoch_validation,\
           _training_split,_validation_split


# 1
def load_data(_dataset,_parameter_dict, _relation_level_words, _device):
    '''load_data'''
    TEMP = data_loading_parameters(_dataset, _parameter_dict)
    _dataset_specific_data_dir, _model_specific_data_dir, _file, _validatation_file, \
    _max_sequence_length, _neg_paths_per_epoch_train, _neg_paths_per_epoch_validation, \
    _training_split, _validation_split = TEMP
    data = {}

    assert _validatation_file is not None
    _training_a = dl.load_data_with_train(_dataset, _dataset_specific_data_dir, _model_specific_data_dir, _file, _max_sequence_length,
                      _neg_paths_per_epoch_train, _relation_level_words, _model='core_chain', _debug=True)
    _valid_a = dl.load_data_with_validation(_dataset, _dataset_specific_data_dir, _model_specific_data_dir, _validatation_file, _max_sequence_length,
                        _neg_paths_per_epoch_validation, _relation_level_words, _model='core_chain', _debug=True)
    print("warning: Test accuracy would not be calculated as the data has not been prepared.")
    data['train_questions'] = _training_a['train_questions']
    data['train_questions_dep'] = _training_a['train_questions_dep']
    data['train_questions_dep_mask'] = _training_a['train_questions_dep_mask']
    data['train_pos_paths'] = _training_a['train_pos_paths']
    data['train_pos_paths_words'] = _training_a['train_pos_paths_words']
    data['train_pos_paths_rel1_sp'] = _training_a['train_pos_paths_rel1_sp']
    data['train_pos_paths_rel2_sp'] = _training_a['train_pos_paths_rel2_sp']
    data['train_pos_paths_rel3_sp'] = _training_a['train_pos_paths_rel3_sp']
    data['train_pos_paths_rel4_sp'] = _training_a['train_pos_paths_rel4_sp']
    data['train_neg_paths'] = _training_a['train_neg_paths']
    data['train_neg_paths_words'] = _training_a['train_neg_paths_words']
    data['train_neg_paths_rel1_sp'] = _training_a['train_neg_paths_rel1_sp']
    data['train_neg_paths_rel2_sp'] = _training_a['train_neg_paths_rel2_sp']
    data['train_neg_paths_rel3_sp'] = _training_a['train_neg_paths_rel3_sp']
    data['train_neg_paths_rel4_sp'] = _training_a['train_neg_paths_rel4_sp']

    data['valid_questions'] = _valid_a['valid_questions']
    data['valid_questions_dep'] = _valid_a['valid_questions_dep']
    data['valid_questions_dep_mask'] = _valid_a['valid_questions_dep_mask']
    data['valid_pos_paths'] = _valid_a['valid_pos_paths']
    data['valid_pos_paths_words'] = _valid_a['valid_pos_paths_words']
    data['valid_pos_paths_rel1_sp'] = _valid_a['valid_pos_paths_rel1_sp']
    data['valid_pos_paths_rel2_sp'] = _valid_a['valid_pos_paths_rel2_sp']
    data['valid_pos_paths_rel3_sp'] = _valid_a['valid_pos_paths_rel3_sp']
    data['valid_pos_paths_rel4_sp'] = _valid_a['valid_pos_paths_rel4_sp']
    data['valid_neg_paths'] = _valid_a['valid_neg_paths']
    data['valid_neg_paths_words'] = _valid_a['valid_neg_paths_words']
    data['valid_neg_paths_rel1_sp'] = _valid_a['valid_neg_paths_rel1_sp']
    data['valid_neg_paths_rel2_sp'] = _valid_a['valid_neg_paths_rel2_sp']
    data['valid_neg_paths_rel3_sp'] = _valid_a['valid_neg_paths_rel3_sp']
    data['valid_neg_paths_rel4_sp'] = _valid_a['valid_neg_paths_rel4_sp']
    data['dummy_y'] = torch.ones(_parameter_dict['batch_size'], device=_device)
    return data


def save_location(problem, model_name, dataset, parameter_dict):
    assert problem in ['core_chain']
    assert dataset in ['graphq','lcquad','compwebq', 'cwq']
    batch_size = parameter_dict['batch_size']
    lr = parameter_dict['learning_rate']
    struct = parameter_dict['struct']
    _neg_paths_per_epoch_train = parameter_dict['_neg_paths_per_epoch_train']
    epochs = parameter_dict['epochs']
    path = 'data/models/' + str(problem)
    if not os.path.exists(path):
        os.makedirs(path)
    dir_name = str(model_name)+'_' +str(dataset) + '_epoch' +str(epochs) +"_bs"+str(batch_size) +"_negs"+str(_neg_paths_per_epoch_train)+"_lr"+str(lr)+"_struct"+str(struct)
    new_path_dir = path + '/' + dir_name
    if not os.path.exists(new_path_dir):
        os.makedirs(new_path_dir)
    return new_path_dir


# Function to save the model
def save_model(loc, modeler, model_name='model.torch', epochs=0, optimizer=None, accuracy=0, aux_save_information={}):
    """
        Input:
            loc: str of the folder where the models are to be saved - data/models/core_chain/cnn_dense_dense/lcquad/5'
            models: dict of 'model_name': model_object
            epochs, optimizers are int, torch.optims (discarded right now).
    """
    state = {
        'epoch': epochs,
        'optimizer': optimizer.state_dict(),
        # 'state_dict': model.state_dict(),
        'accuracy': accuracy
    }
    for tup in modeler.prepare_save():
        state[tup[0]] = tup[1].state_dict()
    aux_save = loc + '/model_info.pickle'
    aux_save_json = loc + '/model_info.json'
    loc = loc + '/' + model_name
    print("model with accuracy ", accuracy, "stored at", loc)
    torch.save(state, loc)
    _aux_save_information = aux_save_information.copy()
    try:
        _aux_save_information['parameter_dict'].pop('vectors')
    except KeyError:
        print("in model save, no vectors were found.")
        pass
    pickle.dump(_aux_save_information,open(aux_save, 'wb+'))
    with open(aux_save_json, 'w', encoding="utf-8") as f:
        json.dump(_aux_save_information, f, indent=4)
    f.close()


import sys

if sys.version_info[0] == 3:
    import configparser as ConfigParser
else:
    import ConfigParser


class Config():
    def __init__(self):
        self.channel_size=8
        self.conv_kernel_1=3
        self.conv_kernel_2=3
        self.pool_kernel_1=3
        self.pool_kernel_2=3
        self.seq_maxlen=25
        self.rel_word_maxlen=25
        self.dropout_prob=0.3


def corechain_parameters(dataset, training_model, config_file ='configs/macros.cfg'):
    config = ConfigParser.ConfigParser()
    config.readfp(open(config_file))
    parameter_dict = {}
    parameter_dict['dataset'] = dataset
    parameter_dict['embedding_dim'] = int(config.get(dataset, 'embedding_dim'))
    parameter_dict['max_length'] = int(config.get('Commons', 'max_length')) #50
    parameter_dict['vocab_size'] = int(config.get(dataset, 'vocab_size'))
    parameter_dict['rel_pad'] = int(config.get(dataset, 'rel_pad')) #25
    parameter_dict['relsp_pad'] = int(config.get(dataset, 'relsp_pad'))  #12
    parameter_dict['_neg_paths_per_epoch_train'] = int(config.get(dataset, '_neg_paths_per_epoch_train'))  #100
    parameter_dict['_neg_paths_per_epoch_validation'] = int(config.get(dataset, '_neg_paths_per_epoch_validation'))  #1000
    parameter_dict['total_negative_samples'] = int(config.get(dataset, 'total_negative_samples'))  #1000
    parameter_dict['batch_size'] = int(config.get(dataset, 'batch_size'))
    parameter_dict['epochs'] = int(config.get('Commons', 'epochs'))
    parameter_dict['hidden_size'] = int(config.get(dataset, 'hidden_size'))  #256
    parameter_dict['number_of_layer'] = int(config.get(dataset, 'number_of_layer'))
    parameter_dict['bidirectional'] = bool(config.get('Commons', 'bidirectional'))
    parameter_dict['dropout'] = float(config.get(dataset, 'dropout'))
    parameter_dict['dropout_rec'] = float(config.get(dataset, 'dropout_rec'))
    parameter_dict['dropout_in'] = float(config.get(dataset, 'dropout_in'))
    if training_model == 'cnn_dot':
        parameter_dict['output_dim'] = int(config.get(dataset, 'output_dim'))
    parameter_dict['learning_rate'] = float(config.get('Commons', 'learning_rate'))
    return parameter_dict



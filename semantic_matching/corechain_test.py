import torch
from fix_seed import set_seed
set_seed(1)
print('#randam:\t', torch.rand(5))

import argparse
from configs import config_loader as cl
from corechain import valid_or_test
import auxiliary as aux


if __name__ == '__main__':
    """test interface"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', action='store', dest='dataset', help='dataset includes lcquad, graphq, cwq', default='lcquad')
    parser.add_argument('-model', action='store', dest='model', help='name of the model to use', default='bert_slotptr_w_dep') #bert_slotptr, bert_slotptr_w_dep
    parser.add_argument('-device', action='store', dest='device', help='cuda for gpu else cpu', default='cpu') #cpu cuda
    parser.add_argument("-struct", type=str, help="The epoches of training", default='skeleton_4260_highest_IR12') #1 20, 30 default='skeleton',
    args = parser.parse_args()
    parameter_dict = cl.corechain_parameters(dataset=args.dataset, training_model=args.model,config_file='configs/macros.cfg')
    parameter_dict['batch_size'] = 64
    parameter_dict['learning_rate'] = 1e-5
    parameter_dict['struct'] = 'skeleton_44979_highest'
    problem = 'core_chain'
    model_save_location = aux.save_location(problem, args.model, args.dataset, parameter_dict)
    print('#model_save_location:\t', model_save_location)
    # valid_acc = valid_or_test(device=args.device, _dataset=args.dataset, training_model=args.model, corechain_model=None, model_save_location=model_save_location, mode='valid')
    test_acc = valid_or_test(device=args.device, _dataset=args.dataset, training_model=args.model, corechain_model=None, model_save_location=model_save_location, mode='test')
    # print('#valid_acc:\t', valid_acc)
    print('#test_acc:\t', test_acc)


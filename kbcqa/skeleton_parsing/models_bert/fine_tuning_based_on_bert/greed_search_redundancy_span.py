import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import model_utils
from fine_tuning_based_on_bert import run_redundancy_span


def grid_search(dataset, learning_rate_list, train_batch_size_list, train_epochs_list):
    folder_list = os.listdir('../tasks/'+dataset+'/debug_redundancy_1222_D/')
    args = model_utils.run_redundancy_span_get_local_args()
    for learning_rate in learning_rate_list:
        for train_batch_size in train_batch_size_list:
            for train_epochs in train_epochs_list:
                args.learning_rate = learning_rate
                args.train_batch_size = train_batch_size
                args.num_train_epochs = train_epochs

                args.bert_model = 'bert-base-cased'
                args.do_train = True
                args.do_predict = True
                args.train_file = '../tasks/'+dataset+'/train-data-new-redundancy.txt'
                args.predict_file = '../tasks/'+dataset+'/dev-data-new-redundancy.txt'
                args.predict_batch_size = 16
                args.max_seq_length = 32
                args.doc_stride = 128
                args.output_dir = '../tasks/'+dataset+'/debug_redundancy_1222_D/' \
                                  'output_train_rate_'+str(learning_rate)+'_batch_'+str(train_batch_size)+'_epochs_'+str(train_epochs)
                print (args.output_dir)
                if 'output_train_rate_'+str(learning_rate)+'_batch_'+str(train_batch_size)+'_epochs_'+str(train_epochs) in folder_list:
                    print('is exist!!!')
                    continue
                run_redundancy_span.main(args=args)


if __name__ == '__main__':
    learning_rate_list = [1e-5, 3e-5, 5e-5]
    train_batch_size_list = [8, 16, 32]
    train_epochs_list = [5,10,20,50,100]

    # dataset = 'fine_tuning_models_lcquad_1217_v0.2'
    # dataset = 'fine_tuning_models_graphq_1222_v0.2'
    # dataset = 'fine_tuning_models_graphq_1228_v0.3'
    dataset = 'fine_tuning_models_cwq_0107_v0.2'
    grid_search(dataset=dataset, learning_rate_list=learning_rate_list, train_batch_size_list=train_batch_size_list, train_epochs_list=train_epochs_list)


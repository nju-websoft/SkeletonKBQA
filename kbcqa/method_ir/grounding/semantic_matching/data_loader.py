import json
import os
import random
import traceback

import numpy as np
import sys
from sklearn.utils import shuffle
import natural_language_utilities as nlutils
import embeddings_interface
import embeddings_interface as ei
from torch.utils.data import Dataset
import math
from auxiliary import read_json

if sys.version_info[0] == 3:
    import configparser as ConfigParser
else:
    import ConfigParser

embeddings_interface.__check_prepared__()
vocabularize_relation_old = lambda path: embeddings_interface.vocabularize(nlutils.tokenize((path))).tolist()
'''__pad__变成了pad'''
special_char_vec = [vocabularize_relation_old(i)for i in embeddings_interface.SPECIAL_CHARACTERS]
vocabularize_relation = lambda path: special_char_vec[embeddings_interface.SPECIAL_CHARACTERS.index(path)]
config = ConfigParser.RawConfigParser()
config.read('configs/macros.cfg')
SEED = config.getint('Commons', 'seed')
NEGATIVE_PATHS_SIZE = config.getint('Commons', 'negative_paths_size')  #cwq 20;  lc 1000, graphq 1000


def remove_positive_path(positive_path, negative_paths,negative_paths_words):
    '''
        Removes positive path from a set of negative paths.
    '''
    counter = 0
    new_negative_paths = []
    new_negative_paths_words = []
    for i in range(0, len(negative_paths)):
        if not np.array_equal(negative_paths[i], positive_path):
            new_negative_paths.append(np.asarray(negative_paths[i]))
            new_negative_paths_words.append(np.asarray(negative_paths_words[i]))
        else:
            counter += 1
            # print counter
    return new_negative_paths,new_negative_paths_words


'''保留正负号'''
def break_path(path,special_chars):
    '''
        Given a path which always starts with special characters . Give two paths.
    :param path:
    :param special_chars:  a list of special characters
    :return:
    '''
    '''看path是一个predicate还是两个predicate'''
    # print('path',path)
    first_sc_index=None
    second_sc_index = None
    third_sc_index = None
    fourth_sc_index = None
    path1=None
    path2=None
    path3=None
    path4=None
    # if (not isinstance(path, np.ndarray)) or isinstance(path, np.int64):
    #     return path, path2, path3, path4
    # if isinstance(path, np.int64):
    #     return path1, path2, path3, path4
    for index,p in enumerate(path):
        if p in special_chars:
            if first_sc_index==None:
                first_sc_index=index
            elif second_sc_index==None:
                second_sc_index=index
            elif third_sc_index==None:
                third_sc_index=index
            elif fourth_sc_index==None:
                fourth_sc_index=index
            else:
                Exception
    if second_sc_index:
        path1=path[first_sc_index:second_sc_index]
        if third_sc_index:
            path2=path[second_sc_index:third_sc_index]
            if fourth_sc_index:
                path3=path[third_sc_index:fourth_sc_index]
                path4=path[fourth_sc_index:]
            else:
                path3=path[third_sc_index:]
        else:
            path2=path[second_sc_index:]
    else:
        path1=path[:] #path[1:]
    # print('path1',path1)
    return path1,path2,path3,path4


# 0
def create_dataset_pairwise(file, max_sequence_length, relation_level_words, _dataset, _dataset_specific_data_dir, _model_specific_data_dir, _model='core_chain'):
    """
            This file is meant to create data for core-chain ranking ONLY.
        :param file:
        :param max_sequence_length: for padding/cropping
        :param relations: the relations file to backtrack and look up shit.
        :return:
        schema decides the kind of data required
            > default - used by all the model apart from slotptr network and reldet
                returns vectors, questions, pos_paths, neg_paths
            >slotptr - used by slot pointer mechanims
                returns vectors, questions, pos_paths, neg_paths
        """
    try:
        with open(os.path.join(_model_specific_data_dir % {'dataset': _dataset, 'model': _model}, file + "_negsize"+ str(NEGATIVE_PATHS_SIZE) + ".mapped.npz"), 'rb') as data:
            dataset = np.load(data, allow_pickle=True)
            ''':k->:-1 最后一个是啥，为啥不要'''
            questions, questions_dep, questions_dep_mask_matrix, \
            pos_paths, neg_paths, \
            pos_paths_rel1_sp, pos_paths_rel2_sp, pos_paths_rel3_sp, pos_paths_rel4_sp,\
            neg_paths_rel1_sp, neg_paths_rel2_sp, neg_paths_rel3_sp, neg_paths_rel4_sp, \
            pos_path_words, neg_paths_words = dataset['arr_0'], dataset['arr_1'], dataset['arr_2'],dataset['arr_3'], \
                                              dataset['arr_4'], dataset['arr_5'], dataset['arr_6'],\
                                              dataset['arr_7'], dataset['arr_8'],dataset['arr_9'], dataset['arr_10'], \
                                              dataset['arr_11'], dataset['arr_12'], dataset['arr_13'], dataset['arr_14']
            vectors = embeddings_interface.vectors
            return vectors, questions, questions_dep, questions_dep_mask_matrix, pos_paths, neg_paths, \
                   pos_paths_rel1_sp, pos_paths_rel2_sp, pos_paths_rel3_sp, pos_paths_rel4_sp,\
                   neg_paths_rel1_sp, neg_paths_rel2_sp, neg_paths_rel3_sp, neg_paths_rel4_sp,\
                   pos_path_words, neg_paths_words
    except (EOFError, IOError) as e:
        with open(os.path.join(_dataset_specific_data_dir % {'dataset': _dataset}, file)) as fp:
            hopnames=['hop3','hop3_0','hop3_1','hop3_2','hop4']
            # hopall=['hop1','hop2','hop3','hop3_0','hop3_1','hop3_2','hop4']
            hopnames.reverse()
            dataset = json.load(fp)
            ignored = []
            dummy_path = [0]
            pos_paths_rel1_sp = []
            pos_paths_rel2_sp = []
            pos_paths_rel3_sp = []
            pos_paths_rel4_sp = []
            neg_paths_rel1_sp = []
            neg_paths_rel2_sp = []
            neg_paths_rel3_sp = []
            neg_paths_rel4_sp = []
            pos_paths = []
            pos_path_words=[]
            for i in dataset:
                # print(i['gold'].keys())
                # if 'path' not in i['gold']:
                #     continue
                try:
                    path_id = i['gold']['path']
                    # if len(path_id)==0:
                    #     ignored.append(i)
                    #     continue
                    # len_candidates=sum([len(i['gold'][key]) for key in hopall if key in i['gold']])
                    # if len_candidates==0:
                    #     ignored.append(i)
                    #     continue
                    positive_path = []
                    positive_path_words=[]
                    for p in path_id:
                        # p = p.lower() lcquad
                        if p in ['+', '-']:
                            positive_path += vocabularize_relation(p)
                        else:
                            positive_path += ei.vocabularize(relation_level_words[p]['0']).tolist()
                            positive_path_words += ei.vocabularize(relation_level_words[p]['0']).tolist()
                except (TypeError, ValueError, KeyError) as e:
                    ignored.append(i)
                    print("error here")
                    print(traceback.print_exc())
                    continue
                pos_paths.append(positive_path)
                pos_path_words.append(positive_path_words)

            questions = [[int(id) for id in list(ei.vocabularize(nlutils.tokenize(i['gold']['abstract_question'].replace('<e>' ,'entity1').replace('<l>', 'literal1'))))]
                            for i in dataset if i not in ignored]
            questions_dep = []
            max_dep_path_length = 2 # 1 0202
            for zz in dataset:
                if zz not in ignored:
                    question_dep = []
                    if 'abstract_question_deppath' in zz['gold']:
                        for abstract_question_deppath_simple in zz['gold']['abstract_question_deppath']:
                            abstract_question_deppath_simple = abstract_question_deppath_simple.replace('<E0>', 'entity1')
                            abstract_question_deppath_simple = abstract_question_deppath_simple.replace('<E1>', 'entity2')
                            abstract_question_deppath_simple = abstract_question_deppath_simple.replace('<E2>', 'entity3')
                            abstract_question_deppath_simple = abstract_question_deppath_simple.replace('<L1>', 'literal1')
                            abstract_question_deppath_simple = abstract_question_deppath_simple.replace('<L2>', 'literal2')
                            abstract_question_deppath_simple = abstract_question_deppath_simple.replace('<L3>', 'literal3')
                            if len(question_dep) < max_dep_path_length:
                                question_dep.append([int(id_) for id_ in list(ei.vocabularize(nlutils.tokenize(abstract_question_deppath_simple.strip())))])
                        """===0202==="""
                        # if len(zz['gold']['abstract_question_deppath']) > max_dep_path_length:
                        #     max_dep_path_length = len(zz['gold']['abstract_question_deppath'])
                        """======"""
                    questions_dep.append(nlutils.pad_sequence(question_dep, max_sequence_length))
            """===question padding==="""
            questions = nlutils.pad_sequence(questions, max_sequence_length)

            """===max dep path length==="""
            print('#max_dep_path_length:\t', max_dep_path_length)
            # if max_dep_path_length > 2: max_dep_path_length = 2
            """======"""

            questions_dep, questions_dep_mask_matrix = nlutils.pad_dependency_sequence(
                allquestions_matrix_seq=questions_dep,
                max_dep_path_length=max_dep_path_length,
                max_length=max_sequence_length)
            """======"""

            neg_paths = []
            neg_paths_words=[]
            for i in range(0, len(pos_paths)):
                if i in ignored: continue
                datum = dataset[i]['gold']
                negative_paths_id=[]
                for key in hopnames:
                    if key in datum:
                        negative_paths_id +=datum[key]
                print(dataset[i]['qid'])
                if 'hop2' in datum:
                    negative_paths_id += (datum['hop2'])
                if 'hop1' in datum:
                    negative_paths_id += ( datum['hop1'])
                # np.random.shuffle(negative_paths_id)
                negative_paths = []
                negative_paths_words=[]
                for neg_path in negative_paths_id:
                    negative_path = []
                    negative_path_words=[]
                    add=True
                    for p in neg_path:
                        # p = p.lower() #lcquad
                        if p in embeddings_interface.SPECIAL_CHARACTERS:
                            negative_path += vocabularize_relation(p)
                        else:
                            if p not in relation_level_words:
                                add=False
                                break
                                # negative_path += ei.vocabularize([p.replace("http://dbpedia.org/property/","")])
                                # print(p)
                            else:
                                if "0" not in relation_level_words[p]:
                                    add=False
                                    break
                                    # print('pppp',p,p.replace("http://dbpedia.org/property/", ""), relation_level_words[p])
                                    # negative_path += ei.vocabularize([p.replace("http://dbpedia.org/property/", "")])
                                else:
                                    negative_path += ei.vocabularize(relation_level_words[p]['0']).tolist()
                                    negative_path_words += ei.vocabularize(relation_level_words[p]['0']).tolist()
                    if add:
                        negative_paths.append(negative_path)
                        negative_paths_words.append(negative_path_words)
                negative_paths, negative_paths_words = remove_positive_path(pos_paths[i], negative_paths, negative_paths_words)
                try:
                    '''选1000个negative paths'''
                    # negative_paths = np.random.choice(negative_paths, 1000)
                    index = np.asarray(random.sample(range(0, len(negative_paths)), NEGATIVE_PATHS_SIZE)) #1000
                    negative_paths = np.array(negative_paths)
                    negative_paths=negative_paths[index]
                    negative_paths_words = np.array(negative_paths_words)
                    negative_paths_words=negative_paths_words[index]
                except ValueError:
                    if len(negative_paths) == 0:
                        try:
                            negative_paths = neg_paths[-1]
                            negative_paths_words = neg_paths_words[-1]
                            print("Using previous question's paths for this since no neg paths for this question.")
                        except IndexError:
                            print("at index error. Moving forward due to a hack")
                            negative_paths = np.asarray([1])
                            negative_paths_words = np.asarray([1])
                    else:
                        '''少于1000就把已有的重复'''
                        index = np.random.randint(0, len(negative_paths), NEGATIVE_PATHS_SIZE)  #1000
                        negative_paths = np.array(negative_paths)
                        negative_paths = negative_paths[index]
                        negative_paths_words=np.array(negative_paths_words)
                        negative_paths_words=negative_paths_words[index]
                neg_paths.append(negative_paths)
                neg_paths_words.append(negative_paths_words)
            special_char = [embeddings_interface.vocabularize(['+']), embeddings_interface.vocabularize(['-'])]

            for pps in pos_paths:
                p1, p2, p3, p4 = break_path(pps, special_char)
                pos_paths_rel1_sp.append(p1)
                if p2 is not None:
                    pos_paths_rel2_sp.append(p2)
                else:
                    pos_paths_rel2_sp.append(dummy_path)
                if p3 is not None:
                    pos_paths_rel3_sp.append(p3)
                else:
                    pos_paths_rel3_sp.append(dummy_path)
                if p4 is not None:
                    pos_paths_rel4_sp.append(p4)
                else:
                    pos_paths_rel4_sp.append(dummy_path)

            for npps in neg_paths:
                temp_neg_paths_rel1_sp = []
                temp_neg_paths_rel2_sp = []
                temp_neg_paths_rel3_sp = []
                temp_neg_paths_rel4_sp = []
                # print('#npps.shape:\t', npps.shape, len(neg_paths))
                for npp in npps:
                    p1, p2, p3, p4 = break_path(npp, special_char)
                    temp_neg_paths_rel1_sp.append(p1)
                    if p2 is not None:
                        temp_neg_paths_rel2_sp.append(p2)
                    else:
                        temp_neg_paths_rel2_sp.append(dummy_path)
                    if p3 is not None:
                        temp_neg_paths_rel3_sp.append(p3)
                    else:
                        temp_neg_paths_rel3_sp.append(dummy_path)
                    if p4 is not None:
                        temp_neg_paths_rel4_sp.append(p4)
                    else:
                        temp_neg_paths_rel4_sp.append(dummy_path)
                neg_paths_rel1_sp.append(temp_neg_paths_rel1_sp)
                neg_paths_rel2_sp.append(temp_neg_paths_rel2_sp)
                neg_paths_rel3_sp.append(temp_neg_paths_rel3_sp)
                neg_paths_rel4_sp.append(temp_neg_paths_rel4_sp)

            for i in range(0, len(neg_paths)):
                neg_paths[i] = nlutils.pad_sequence(neg_paths[i], max_sequence_length)
                neg_paths_words[i] = nlutils.pad_sequence(neg_paths_words[i], max_sequence_length)
                neg_paths_rel1_sp[i] = nlutils.pad_sequence(neg_paths_rel1_sp[i], max_sequence_length)
                neg_paths_rel2_sp[i] = nlutils.pad_sequence(neg_paths_rel2_sp[i], max_sequence_length)
                neg_paths_rel3_sp[i] = nlutils.pad_sequence(neg_paths_rel3_sp[i], max_sequence_length)
                neg_paths_rel4_sp[i] = nlutils.pad_sequence(neg_paths_rel4_sp[i], max_sequence_length)
            neg_paths = np.asarray(neg_paths)
            neg_paths_words = np.asarray(neg_paths_words)
            neg_paths_rel1_sp = np.asarray(neg_paths_rel1_sp)
            neg_paths_rel2_sp = np.asarray(neg_paths_rel2_sp)
            neg_paths_rel3_sp = np.asarray(neg_paths_rel3_sp)
            neg_paths_rel4_sp = np.asarray(neg_paths_rel4_sp)

            pos_paths = nlutils.pad_sequence(pos_paths, max_sequence_length)
            pos_path_words = nlutils.pad_sequence(pos_path_words, max_sequence_length)
            pos_paths_rel1_sp = nlutils.pad_sequence(pos_paths_rel1_sp, max_sequence_length)
            pos_paths_rel2_sp = nlutils.pad_sequence(pos_paths_rel2_sp, max_sequence_length)
            pos_paths_rel3_sp = nlutils.pad_sequence(pos_paths_rel3_sp, max_sequence_length)
            pos_paths_rel4_sp = nlutils.pad_sequence(pos_paths_rel4_sp, max_sequence_length)
            vectors = embeddings_interface.vectors
            print("att ht place where things are made")
            with open(os.path.join(_model_specific_data_dir % {'dataset': _dataset, 'model': _model}, file + "_negsize"+ str(NEGATIVE_PATHS_SIZE) + ".mapped.npz"), "wb") as data:
                np.savez(data, questions, questions_dep, questions_dep_mask_matrix, pos_paths, neg_paths,
                         pos_paths_rel1_sp, pos_paths_rel2_sp,pos_paths_rel3_sp, pos_paths_rel4_sp,
                         neg_paths_rel1_sp, neg_paths_rel2_sp, neg_paths_rel3_sp, neg_paths_rel4_sp,
                         pos_path_words, neg_paths_words)
            return vectors, questions, questions_dep, questions_dep_mask_matrix, pos_paths, neg_paths, \
                   pos_paths_rel1_sp, pos_paths_rel2_sp, pos_paths_rel3_sp, pos_paths_rel4_sp,\
                   neg_paths_rel1_sp, neg_paths_rel2_sp, neg_paths_rel3_sp, neg_paths_rel4_sp,\
                   pos_path_words, neg_paths_words


# 2
def load_data_with_train(_dataset, _dataset_specific_data_dir, _model_specific_data_dir, _file, _max_sequence_length,
                          _neg_paths_per_epoch_train, _relation_level_words, _model='core_chain', _debug=True):
    '''load_data'''
    vectors, questions, question_dep, questions_dep_mask_matrix,\
    pos_paths, neg_paths, \
    pos_paths_rel1_sp, pos_paths_rel2_sp, pos_paths_rel3_sp, pos_paths_rel4_sp, \
    neg_paths_rel1_sp, neg_paths_rel2_sp, neg_paths_rel3_sp, neg_paths_rel4_sp, \
    pos_path_words, neg_paths_words = create_dataset_pairwise(file=_file, max_sequence_length=_max_sequence_length,
                                                              relation_level_words=_relation_level_words,
                                                              _dataset=_dataset,
                                                              _dataset_specific_data_dir=_dataset_specific_data_dir,
                                                              _model_specific_data_dir=_model_specific_data_dir,
                                                              _model='core_chain')
    data = {}
    for i in range(0, len(pos_paths)):
        for j in range(0, len(neg_paths[i])):
            if np.array_equal(pos_paths[i], neg_paths[i][j]):
                if j == 0:
                    neg_paths[i][j] = neg_paths[i][j + 10]
                    neg_paths_words[i][j] = neg_paths_words[i][j + 10]
                    neg_paths_rel1_sp[i][j] = neg_paths_rel1_sp[i][j + 10]
                    neg_paths_rel2_sp[i][j] = neg_paths_rel2_sp[i][j + 10]
                    neg_paths_rel3_sp[i][j] = neg_paths_rel3_sp[i][j + 10]
                    neg_paths_rel4_sp[i][j] = neg_paths_rel4_sp[i][j + 10]
                else:
                    neg_paths[i][j] = neg_paths[i][0]
                    neg_paths_words[i][j] = neg_paths_words[i][0]
                    neg_paths_rel1_sp[i][j] = neg_paths_rel1_sp[i][0]
                    neg_paths_rel2_sp[i][j] = neg_paths_rel2_sp[i][0]
                    neg_paths_rel3_sp[i][j] = neg_paths_rel3_sp[i][0]
                    neg_paths_rel4_sp[i][j] = neg_paths_rel4_sp[i][0]
    entity = questions
    data['train_pos_paths'] = pos_paths
    data['train_pos_paths_words'] = pos_path_words
    data['train_pos_paths_rel1_sp'] = pos_paths_rel1_sp
    data['train_pos_paths_rel2_sp'] = pos_paths_rel2_sp
    data['train_pos_paths_rel3_sp'] = pos_paths_rel3_sp
    data['train_pos_paths_rel4_sp'] = pos_paths_rel4_sp
    data['train_neg_paths'] = neg_paths
    data['train_neg_paths_words'] = neg_paths_words
    data['train_neg_paths_rel1_sp'] = neg_paths_rel1_sp
    data['train_neg_paths_rel2_sp'] = neg_paths_rel2_sp
    data['train_neg_paths_rel3_sp'] = neg_paths_rel3_sp
    data['train_neg_paths_rel4_sp'] = neg_paths_rel4_sp
    data['train_questions'] = questions
    data['train_questions_dep'] = question_dep
    data['train_questions_dep_mask'] = questions_dep_mask_matrix
    data['train_entity'] = entity
    data['dummy_y_train'] = np.zeros(len(data['train_questions']) * _neg_paths_per_epoch_train)
    # data['dummy_y_valid'] = np.zeros(len(data['valid_questions']) * (_neg_paths_per_epoch_validation + 1))  # 为什么加1
    # 上面这两位没有用到啊
    data['vectors'] = vectors
    return data

# 3
def load_data_with_validation(_dataset, _dataset_specific_data_dir, _model_specific_data_dir, _file, _max_sequence_length,
                          _neg_paths_per_epoch_validation, _relation_level_words, _model='core_chain', _debug=True):
    '''load_data'''
    vectors, questions, question_dep, questions_dep_mask_matrix, pos_paths, neg_paths, \
    pos_paths_rel1_sp, pos_paths_rel2_sp, pos_paths_rel3_sp, pos_paths_rel4_sp, \
    neg_paths_rel1_sp, neg_paths_rel2_sp, neg_paths_rel3_sp, neg_paths_rel4_sp, \
    pos_path_words, neg_paths_words = create_dataset_pairwise(file=_file, max_sequence_length=_max_sequence_length,
                                                              relation_level_words=_relation_level_words,
                                                              _dataset=_dataset,
                                                              _dataset_specific_data_dir=_dataset_specific_data_dir,
                                                              _model_specific_data_dir=_model_specific_data_dir,
                                                              _model='core_chain')
    data = {}
    for i in range(0, len(pos_paths)):
        for j in range(0, len(neg_paths[i])):
            if np.array_equal(pos_paths[i], neg_paths[i][j]):
                if j == 0:
                    neg_paths[i][j] = neg_paths[i][j + 10]
                    neg_paths_words[i][j] = neg_paths_words[i][j + 10]
                    neg_paths_rel1_sp[i][j] = neg_paths_rel1_sp[i][j + 10]
                    neg_paths_rel2_sp[i][j] = neg_paths_rel2_sp[i][j + 10]
                    neg_paths_rel3_sp[i][j] = neg_paths_rel3_sp[i][j + 10]
                    neg_paths_rel4_sp[i][j] = neg_paths_rel4_sp[i][j + 10]
                else:
                    neg_paths[i][j] = neg_paths[i][0]
                    neg_paths_words[i][j] = neg_paths_words[i][0]
                    neg_paths_rel1_sp[i][j] = neg_paths_rel1_sp[i][0]
                    neg_paths_rel2_sp[i][j] = neg_paths_rel2_sp[i][0]
                    neg_paths_rel3_sp[i][j] = neg_paths_rel3_sp[i][0]
                    neg_paths_rel4_sp[i][j] = neg_paths_rel4_sp[i][0]
    entity = questions
    data['valid_pos_paths'] = pos_paths
    data['valid_pos_paths_words'] = pos_path_words
    data['valid_pos_paths_rel1_sp'] = pos_paths_rel1_sp
    data['valid_pos_paths_rel2_sp'] = pos_paths_rel2_sp
    data['valid_pos_paths_rel3_sp'] = pos_paths_rel3_sp
    data['valid_pos_paths_rel4_sp'] = pos_paths_rel4_sp
    data['valid_neg_paths'] = neg_paths
    data['valid_neg_paths_words'] = neg_paths_words
    data['valid_neg_paths_rel1_sp'] = neg_paths_rel1_sp
    data['valid_neg_paths_rel2_sp'] = neg_paths_rel2_sp
    data['valid_neg_paths_rel3_sp'] = neg_paths_rel3_sp
    data['valid_neg_paths_rel4_sp'] = neg_paths_rel4_sp
    data['valid_questions'] = questions
    data['valid_questions_dep'] = question_dep
    data['valid_questions_dep_mask'] = questions_dep_mask_matrix
    data['valid_entity'] = entity
    data['dummy_y_valid'] = np.zeros(len(data['valid_questions']) * (_neg_paths_per_epoch_validation + 1))  # 为什么加1
    # 上面这两位没有用到啊
    data['vectors'] = vectors
    return data


class TrainingDataGenerator(Dataset):

    def __init__(self, data, max_length, neg_paths_per_epoch, batch_size,total_negative_samples,schema='default',snip=1.0):
        self.dummy_y = np.zeros(batch_size)
        self.firstDone = False
        self.max_length = max_length
        self.neg_paths_per_epoch = neg_paths_per_epoch
        self.total_negative_samples = total_negative_samples
        self.schema = schema

        def snipper(d):
            return int(len(d)*snip)

        questions = data['train_questions'][:snipper(data['train_questions'])]
        questions_dep = data['train_questions_dep'][:snipper(data['train_questions_dep'])]
        questions_dep_mask = data['train_questions_dep_mask'][:snipper(data['train_questions_dep_mask'])]

        pos_paths = data['train_pos_paths'][:snipper(data['train_pos_paths'])]
        pos_paths_words = data['train_pos_paths_words'][:snipper(data['train_pos_paths_words'])]

        neg_paths = data['train_neg_paths'][:snipper(data['train_neg_paths'])]
        neg_paths_words = data['train_neg_paths_words'][:snipper(data['train_neg_paths_words'])]

        if schema == 'slotptr':
            self.pos_paths_rel1 = data['train_pos_paths_rel1_sp'][:snipper(data['train_pos_paths_rel1_sp'])]
            self.pos_paths_rel2 = data['train_pos_paths_rel2_sp'][:snipper(data['train_pos_paths_rel2_sp'])]
            self.pos_paths_rel3 = data['train_pos_paths_rel3_sp'][:snipper(data['train_pos_paths_rel3_sp'])]
            self.pos_paths_rel4 = data['train_pos_paths_rel4_sp'][:snipper(data['train_pos_paths_rel4_sp'])]

            self.neg_paths_rel1 = data['train_neg_paths_rel1_sp'][:snipper(data['train_neg_paths_rel1_sp'])]
            self.neg_paths_rel2 = data['train_neg_paths_rel2_sp'][:snipper(data['train_neg_paths_rel2_sp'])]
            self.neg_paths_rel3 = data['train_neg_paths_rel3_sp'][:snipper(data['train_neg_paths_rel3_sp'])]
            self.neg_paths_rel4 = data['train_neg_paths_rel4_sp'][:snipper(data['train_neg_paths_rel4_sp'])]
        # print(questions.shape)
        self.questions = np.reshape(np.repeat((np.reshape(questions, (questions.shape[0], 1, questions.shape[1]))), neg_paths_per_epoch, axis=1), (-1, max_length))
        print(questions_dep.shape)
        self.questions_dep = np.repeat(questions_dep, neg_paths_per_epoch, axis=0)
        # (np.reshape(questions_dep, (questions_dep.shape[0], -1, questions_dep.shape[2]))),
        # self.questions_dep = np.reshape(questions_dep_repeat,(-1, questions_dep.shape[1], max_length))
        # self.questions_dep = np.reshape(np.repeat((np.reshape(questions_dep, (questions_dep.shape[0], 1, questions_dep.shape[1]))), neg_paths_per_epoch, axis=1), (-1, max_length))
        self.questions_dep_mask = np.reshape(np.repeat((np.reshape(questions_dep_mask, (questions_dep_mask.shape[0], 1, questions_dep_mask.shape[1]))), neg_paths_per_epoch, axis=1), (-1, questions_dep_mask.shape[1]))

        self.pos_paths = np.reshape(np.repeat(np.reshape(pos_paths, (pos_paths.shape[0], 1, pos_paths.shape[1])), neg_paths_per_epoch, axis=1), (-1, max_length))
        self.pos_paths_words = np.reshape(np.repeat(np.reshape(pos_paths_words, (pos_paths_words.shape[0], 1, pos_paths_words.shape[1])),neg_paths_per_epoch, axis=1), (-1, max_length))

        self.neg_paths = neg_paths
        self.neg_paths_words = neg_paths_words
        sampling_index = np.random.randint(0, self.total_negative_samples, self.neg_paths_per_epoch)
        self.neg_paths_sampled = np.reshape(self.neg_paths[:, sampling_index, :], (-1, self.max_length))
        self.neg_paths_words_sampled = np.reshape(self.neg_paths_words[:, sampling_index, :],(-1, self.max_length))
        if self.schema != 'default':
            self.neg_paths_rel1_sampled = np.reshape(self.neg_paths_rel1[:, sampling_index, :],(-1, self.max_length))
            self.neg_paths_rel2_sampled = np.reshape(self.neg_paths_rel2[:, sampling_index, :],(-1, self.max_length))
            self.neg_paths_rel3_sampled = np.reshape(self.neg_paths_rel3[:, sampling_index, :],(-1, self.max_length))
            self.neg_paths_rel4_sampled = np.reshape(self.neg_paths_rel4[:, sampling_index, :],(-1, self.max_length))

            self.pos_paths_rel1 = np.reshape(np.repeat(np.reshape(self.pos_paths_rel1, (self.pos_paths_rel1.shape[0], 1, self.pos_paths_rel1.shape[1])), self.neg_paths_per_epoch, axis=1), (-1, self.max_length))
            self.pos_paths_rel2 = np.reshape(np.repeat(np.reshape(self.pos_paths_rel2, (self.pos_paths_rel2.shape[0], 1, self.pos_paths_rel2.shape[1])), self.neg_paths_per_epoch, axis=1), (-1, self.max_length))
            self.pos_paths_rel3 = np.reshape(np.repeat(np.reshape(self.pos_paths_rel3, (self.pos_paths_rel3.shape[0], 1, self.pos_paths_rel3.shape[1])), self.neg_paths_per_epoch, axis=1), (-1, self.max_length))
            self.pos_paths_rel4 = np.reshape(np.repeat(np.reshape(self.pos_paths_rel4, (self.pos_paths_rel4.shape[0], 1, self.pos_paths_rel4.shape[1])), self.neg_paths_per_epoch, axis=1), (-1, self.max_length))

        if schema == 'default':
            self.questions_shuffled, self.questions_dep_shuffled, self.questions_dep_mask_shuffled,\
            self.pos_paths_shuffled, self.pos_paths_words_shuffled, \
            self.neg_paths_shuffled, self.neg_paths_words_shuffled = \
            shuffle(self.questions, self.questions_dep, self.questions_dep_mask,
                    self.pos_paths, self.pos_paths_words,
                    self.neg_paths_sampled,self.neg_paths_words_sampled)
        else:
            self.questions_shuffled, self.questions_dep_shuffled, self.questions_dep_mask_shuffled, \
            self.pos_paths_shuffled, self.pos_paths_rel1_shuffled, self.pos_paths_rel2_shuffled, self.pos_paths_rel3_shuffled, self.pos_paths_rel4_shuffled,\
            self.neg_paths_shuffled, self.neg_paths_rel1_shuffled, self.neg_paths_rel2_shuffled,self.neg_paths_rel3_shuffled,self.neg_paths_rel4_shuffled , \
            self.pos_paths_words_shuffled,self.neg_paths_words_shuffled = \
                shuffle(self.questions, self.questions_dep, self.questions_dep_mask,
                        self.pos_paths, self.pos_paths_rel1, self.pos_paths_rel2,self.pos_paths_rel3,self.pos_paths_rel4,
                        self.neg_paths_sampled,self.neg_paths_rel1_sampled, self.neg_paths_rel2_sampled,self.neg_paths_rel3_sampled, self.neg_paths_rel4_sampled,
                        self.pos_paths_words, self.neg_paths_words_sampled)
        self.batch_size = batch_size

    def __len__(self):
        print(len(self.questions))
        return math.ceil(len(self.questions) / self.batch_size)

    def __getitem__(self, idx):
        """
            Called at every iter.
            If code not pointwise, simple sample (not randomly) next batch items.
            If pointwise:
                you use the same sampled things, only that you then concat neg and pos paths,
                and subsample half from there.
        :param idx:
        :return:
        """
        index = lambda x: x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_questions = index(self.questions_shuffled)    # Shape (batch, seqlen)
        batch_questions_dep = index(self.questions_dep_shuffled)  #Shape (batch, seqlen)
        batch_questions_dep_mask = index(self.questions_dep_mask_shuffled)  #Shape (batch, seqlen)
        batch_pos_paths = index(self.pos_paths_shuffled)    # Shape (batch, seqlen)
        batch_pos_paths_words = index(self.pos_paths_words_shuffled)    # Shape (batch, seqlen)
        batch_neg_paths = index(self.neg_paths_shuffled)    # Shape (batch, seqlen)
        batch_neg_paths_words = index(self.neg_paths_words_shuffled)    # Shape (batch, seqlen)
        if self.schema != 'default':
            batch_neg_paths_rel1 = index(self.neg_paths_rel1_shuffled)  # Shape (batch, seqlen)
            batch_neg_paths_rel2 = index(self.neg_paths_rel2_shuffled)  # Shape (batch, seqlen)
            batch_neg_paths_rel3 = index(self.neg_paths_rel3_shuffled)  # Shape (batch, seqlen)
            batch_neg_paths_rel4 = index(self.neg_paths_rel4_shuffled)  # Shape (batch, seqlen)
            batch_pos_paths_rel1 = index(self.pos_paths_rel1_shuffled)  # Shape (batch, seqlen)
            batch_pos_paths_rel2 = index(self.pos_paths_rel2_shuffled)  # Shape (batch, seqlen)
            batch_pos_paths_rel3 = index(self.pos_paths_rel3_shuffled)  # Shape (batch, seqlen)
            batch_pos_paths_rel4 = index(self.pos_paths_rel4_shuffled)  # Shape (batch, seqlen)

        if self.schema == 'default':
            return ([batch_questions, batch_questions_dep, batch_questions_dep_mask,
                     batch_pos_paths, batch_neg_paths,
                     batch_pos_paths_words,batch_neg_paths_words], self.dummy_y)
        else:
            return ([batch_questions, batch_questions_dep, batch_questions_dep_mask,
                     batch_pos_paths,
                     batch_pos_paths_rel1,batch_pos_paths_rel2,batch_pos_paths_rel3,batch_pos_paths_rel4,
                     batch_neg_paths,
                     batch_neg_paths_rel1,batch_neg_paths_rel2,batch_neg_paths_rel3,batch_neg_paths_rel4,
                     batch_pos_paths_words,batch_neg_paths_words], self.dummy_y)

    def shuffle(self):
        """
            To be called at the end of every epoch. We sample new negative paths,
            \and then we shuffle the questions, pos and neg paths in tandem.
        :return: None
        """
        sampling_index = np.random.randint(0, self.total_negative_samples, self.neg_paths_per_epoch)
        self.neg_paths_sampled = np.reshape(self.neg_paths[:, sampling_index , :], (-1, self.max_length))
        self.neg_paths_words_sampled = np.reshape(self.neg_paths_words[:, sampling_index, :], (-1, self.max_length))
        if self.schema != 'default':
            self.neg_paths_rel1_sampled = np.reshape(self.neg_paths_rel1[:, sampling_index, :],(-1, self.max_length))
            self.neg_paths_rel2_sampled = np.reshape(self.neg_paths_rel2[:, sampling_index, :],(-1, self.max_length))
            self.neg_paths_rel3_sampled = np.reshape(self.neg_paths_rel3[:, sampling_index, :],(-1, self.max_length))
            self.neg_paths_rel4_sampled = np.reshape(self.neg_paths_rel4[:, sampling_index, :],(-1, self.max_length))

            self.questions_shuffled, self.questions_dep_shuffled, self.questions_dep_mask_shuffled,\
            self.pos_paths_shuffled, \
            self.pos_paths_rel1_shuffled,self.pos_paths_rel2_shuffled, self.pos_paths_rel3_shuffled, self.pos_paths_rel4_shuffled, \
            self.neg_paths_shuffled, \
            self.neg_paths_rel1_shuffled, self.neg_paths_rel2_shuffled,self.neg_paths_rel3_shuffled,self.neg_paths_rel4_shuffled,\
            self.pos_paths_words_shuffled, self.neg_paths_words_shuffled = \
                shuffle(self.questions, self.questions_dep, self.questions_dep_mask,
                        self.pos_paths,
                        self.pos_paths_rel1, self.pos_paths_rel2,self.pos_paths_rel3,self.pos_paths_rel4,
                        self.neg_paths_sampled,
                        self.neg_paths_rel1_sampled, self.neg_paths_rel2_sampled, self.neg_paths_rel3_sampled, self.neg_paths_rel4_sampled,
                        self.pos_paths_words, self.neg_paths_words_sampled)
        else:
            self.questions_shuffled, self.questions_dep_shuffled, self.questions_dep_mask_shuffled,\
            self.pos_paths_shuffled, self.neg_paths_shuffled,self.pos_paths_words_shuffled, self.neg_paths_words_shuffled = \
                shuffle(self.questions, self.questions_dep, self.questions_dep_mask,
                        self.pos_paths, self.neg_paths_sampled,self.pos_paths_words,self.neg_paths_words_sampled)


def create_dataset_runtime(file,_dataset_specific_data_dir):
    '''Function loads the data from the _dataset_specific_data_dir+ file and splits it in case of LCQuAD'''
    id_data_test=read_json(os.path.join(_dataset_specific_data_dir, file))
    vectors = embeddings_interface.vectors
    return id_data_test, vectors


def construct_paths(data, relation_level_words, qald=False,goldorpred='gold'):
    """
        For a given datanode, the function constructs positive and negative paths and prepares question uri.
        :param data: a data node of id_big_data
        relations : a dictionary which maps relation id to meta inforamtion like surface form, embedding id
        of surface form etc.
        :return: unpadded , continous id spaced question, positive path, negative paths
    """
    abstract_question = data[goldorpred]['abstract_question'].replace('<e>', 'entity1').replace('<l>', 'literal1')
    question = ei.vocabularize(nlutils.tokenize(abstract_question))

    """======"""
    question_dep = []
    if 'abstract_question_deppath' in data['gold']:
        for abstract_question_deppath_simple in data['gold']['abstract_question_deppath']:
            abstract_question_deppath_simple = abstract_question_deppath_simple.replace('<E0>', 'entity1')
            abstract_question_deppath_simple = abstract_question_deppath_simple.replace('<E1>', 'entity2')
            abstract_question_deppath_simple = abstract_question_deppath_simple.replace('<E2>', 'entity3')
            abstract_question_deppath_simple = abstract_question_deppath_simple.replace('<L1>', 'literal1')
            abstract_question_deppath_simple = abstract_question_deppath_simple.replace('<L2>', 'literal2')
            abstract_question_deppath_simple = abstract_question_deppath_simple.replace('<L3>', 'literal3')
            question_dep.append([int(id_) for id_ in list(
                ei.vocabularize(nlutils.tokenize(abstract_question_deppath_simple.strip())))])
    if len(question_dep) == 0:
        question_dep.append([int(id_) for id_ in list(ei.vocabularize(nlutils.tokenize(abstract_question.strip())))])
    question_dep_mask_matrix = 1.0*np.ones((1, len(question_dep)))

    """======"""

    '''goldpathindex 可能要用于mrr计算,有了goldpathindex其实就不需要no_positive_path'''
    candidates=[]
    for key in ['hop4','hop3_2','hop3_1','hop3_0','hop3','hop2','hop1']:
        if key in data[goldorpred]:
            candidates+=data[goldorpred][key]

    ####get gold path####
    goldpathindex = -1
    for index,candidate in enumerate(candidates):
        if np.array_equal(candidate, data['gold']['path']):
            goldpathindex=index
            break

    ##########get candidate path#####
    candidate_paths = []
    candidate_paths_words = []
    for cand_path in candidates:
        candidate_path=[]
        candidate_path_words=[]
        add=True
        for p in cand_path:
            # p = p.lower() lcquad
            if p in embeddings_interface.SPECIAL_CHARACTERS:
                candidate_path.extend( vocabularize_relation(p))
            else:
                if p not in relation_level_words:
                    # add=False
                    # break
                    candidate_path.extend( ei.vocabularize([p.replace("http://dbpedia.org/property/", "")]))
                    candidate_path_words.extend( ei.vocabularize([p.replace("http://dbpedia.org/property/", "")]))
                else:
                    if "0" not in relation_level_words[p]:
                        # add=False
                        # break
                        # print('pppp', p, p.replace("http://dbpedia.org/property/", ""), relation_level_words[p])
                        candidate_path.extend( ei.vocabularize([p.replace("http://dbpedia.org/property/", "")]))
                        # print('before',candidate_path_words)
                        candidate_path_words.extend( ei.vocabularize([p.replace("http://dbpedia.org/property/", "")]))
                        # print('end',ei.vocabularize([p.replace("http://dbpedia.org/property/", "")]),candidate_path_words)
                    else:
                        candidate_path.extend( ei.vocabularize(relation_level_words[p]['0']).tolist())
                        candidate_path_words.extend( ei.vocabularize(relation_level_words[p]['0']).tolist())
        if add:
            candidate_paths.append(np.asarray(candidate_path))
            candidate_paths_words.append(np.asarray(candidate_path_words))

    return question,\
           np.asarray(question_dep), np.asarray(question_dep_mask_matrix),\
           np.asarray(candidate_paths), np.asarray(candidate_paths_words),\
           goldpathindex, candidates


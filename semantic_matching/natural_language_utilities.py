
import numpy as np
import re


def tokenize(_input, _ignore_brackets=False, _remove_stopwords=False):
    """
        Tokenize a question.
        Changes:
            - removes question marks
            - removes commas
            - removes trailing spaces
            - can remove text inside one-level brackets.
        @TODO: Improve tokenization
        Used in: parser.py; krantikari.py
        :param _input: str,
        :return: list of tokens
    """
    cleaner_input = _input.replace("?", "").replace(",", "").strip()
    if _ignore_brackets:
        # If there's some text b/w brackets, remove it. @TODO: NESTED parenthesis not covered.
        pattern = r'\([^\)]*\)'
        matcher = re.search(pattern, cleaner_input, 0)
        if matcher:
            substring = matcher.group()
            cleaner_input = cleaner_input[:cleaner_input.index(substring)] + cleaner_input[cleaner_input.index(substring) + len(substring):]
    return cleaner_input.strip().split() if not _remove_stopwords else Exception


def pad_sequence(matrix_seq,max_length):
    '''
    #Works with list od list as well as numpy matrix
    :param sequence: a matrix of list
    :param max_length:
    :return:
    '''
    pad_matrix = np.zeros((len(matrix_seq), max_length))
    for i, arr in enumerate(matrix_seq):
        pad_matrix[i, :min(max_length, len(arr))] = arr[:min(max_length, len(arr))]
    return pad_matrix


def pad_dependency_sequence(allquestions_matrix_seq, max_dep_path_length, max_length):
    '''
    max length padding
    mask vector: 1 represent mask, 0 contain value
    '''
    mask_matrix = np.zeros((len(allquestions_matrix_seq), max_dep_path_length))
    padding_result = []
    for i, onequestions_matraix_seq in enumerate(allquestions_matrix_seq):
        pad_matrix = np.zeros((max_dep_path_length, max_length))  #2, 20
        for j, arr in enumerate(onequestions_matraix_seq):
            pad_matrix[j, :min(max_length, len(arr))] = arr[:min(max_length, len(arr))]
        # for j in range(len(onequestions_matraix_seq), max_dep_path_length):
        #     pad_matrix[j, :] = pad_matrix[j-1,:]
        padding_result.append(pad_matrix)
        mask_matrix[i,:] = 1.0*~(np.arange(max_dep_path_length) >= len(onequestions_matraix_seq))
    padding_result = np.array(padding_result, dtype='float64')
    return padding_result, mask_matrix


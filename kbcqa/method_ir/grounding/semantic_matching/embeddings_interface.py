import traceback
import warnings
import numpy as np
import pickle
import os

DEBUG = True
vectors, vocab, temp_vocab_id = [], {}, {'_pad_':(0,False), '_unk_':(1,False), '+':(2,False), '-':(3,False), '/':(4,False), 'uri':(5,False), 'x':(6,False)}
# vectors, vocab = [], {}

POSSIBLE_EMBEDDINGS = ['glove', 'fasttext', 'ulmfit']
DEFAULT_EMBEDDING = POSSIBLE_EMBEDDINGS[0]
SELECTED_EMBEDDING = None
SPECIAL_CHARACTERS = ['_pad_', '_unk_', '+', '-', '/', 'uri', 'x']
SPECIAL_EMBEDDINGS = [0, 0, 1, -1, 0.5, -2, 2]
# SPECIAL_CHARACTERS = []
# SPECIAL_EMBEDDINGS = []
####################deplabel########################
import json
with open('./resources/deplabel.json', 'r', encoding="utf-8") as f:
    dep_dict = json.load(f)
f.close()
for dep,value in dep_dict.items():
    SPECIAL_CHARACTERS.append(dep)
    SPECIAL_EMBEDDINGS.append(value)
####################################################

GLOVE_LENGTH = 2196017
EMBEDDING_DIM = 400
EMBEDDING_GLOVE_DIM = 300
EMBEDDING_FASTAI_DIM = 300  # @TODO: fix
PREPARED = False
parsed_location = './resources'
glove_location = \
    {
        'dir': "./resources",
        'raw': "glove.42B.300d.txt",
        'vec': "vectors_gl.npy",
        'voc': "vocab_gl.pickle"
    }

def __check_prepared__(_embedding=None):
    if len(vectors) <= len(SPECIAL_CHARACTERS) or len(vocab) <= len(SPECIAL_CHARACTERS) or (_embedding != None and _embedding != SELECTED_EMBEDDING):
        __prepare__(_embedding)

def _init_special_characters_():
    """
        Regardless of whatever we choose, vectors and vocab need to have basic stuff in them.
        Depends on what we mention as special characters.
        This fn assumes empty vector, vocab
    """
    global vectors, vocab
    try:
        assert len(vectors) == 0 & len(vocab) == 0
    except AssertionError:
        warnings.warn("Found non empty vectors, vocab. Cleaning them up.")
        for sp_char in SPECIAL_CHARACTERS:
            assert sp_char in vocab
    # Push special chars in the vocab, alongwith their vectors IF not already there.
    for i, sp_char in enumerate(SPECIAL_CHARACTERS):
        vocab[sp_char] = i
        vectors.append(np.repeat(SPECIAL_EMBEDDINGS[i], EMBEDDING_DIM))

def __prepare__(_embedding=None):
    global SELECTED_EMBEDDING, EMBEDDING_DIM
    # If someone gave an embedding, mark that as the permanent one
    SELECTED_EMBEDDING = _embedding if _embedding != None else DEFAULT_EMBEDDING
    EMBEDDING_DIM = 300 if SELECTED_EMBEDDING in ['glove'] else 400
    _init_special_characters_()
    if SELECTED_EMBEDDING == 'glove':
        _parse_glove_()

def load(_embedding):
    if _embedding == 'glove':
        locs = glove_location
    local_vocab = pickle.load(open(os.path.join(parsed_location, locs['voc']), 'rb'))
    local_vectors = np.load(os.path.join(parsed_location, locs['vec']))
    return local_vocab, local_vectors

def __parse_line__(line):
    """
        Used for glove raw file parsing.
        Partitions the list into two depending on till where in it do words exist.
        e.g. the 1 2 3 will be 'the' [1 2 3]
        eg. the person 1 2 will be 'the person' [1 2]
    """
    tokens = line.split(' ')
    tokens[-1] = tokens[-1][:-1]
    word = [tokens[0]]
    tokens = [float(t) for  t in tokens[1:]]
    while True:
        token = tokens[0]
        #         print(tokreen)
        try:
            _ = float(token)
            break
        except ValueError:
            word.append(token)
            tokens.pop(0)
    #     print(line)
    assert len(tokens) == 300  # Hardcoded here, because we know glove pretrained is 300d
    #     raise EOFError
    return ' '.join(word), np.asarray(tokens)

def save():
    if SELECTED_EMBEDDING == 'glove':
        locs = glove_location
    if DEBUG:
        print("Saving %(emb)s in %(loc)s" % {'emb': SELECTED_EMBEDDING, 'loc': parsed_location})
    # Save vectors
    np.save(os.path.join(parsed_location, locs['vec']), vectors)
    # Save vocab
    pickle.dump(vocab, open(os.path.join(parsed_location, locs['voc']), 'wb+'))

def _parse_glove_():
    """
        Fn to go through glove's raw file, and add vocab, vectors for words not already in vocab.
    """
    global vectors, vocab
    print("Loading Glove vocab and vectors from disk. Sit Tight.")
    try:
        # Try to load from disk
        vocab, vectors = load(_embedding='glove')
        return True
    except FileNotFoundError:
        warnings.warn("Couldn't find Glove vocab and/or vectors on disk. Parsing from raw file will TAKE TIME ...")
        # Assume that vectors can be list OR numpy array.
        changes = 0
        lines = 0
        new_vectors = []
        # Open raw file
        f = open(os.path.join(glove_location['dir'], glove_location['raw']), encoding='utf-8')
        for line in f:
            lines += 1
            # Parse line
            word, coefs = __parse_line__(line)
            # Ignore if word is a special char
            if word in SPECIAL_CHARACTERS:
                continue
            # Ignore if we already have this word
            try:
                _ = vocab[word]
                continue
            except KeyError:
                # Its a new word, put it in somewhere.
                vocab[word] = len(vocab)
                new_vectors.append(coefs)
                changes += 1
        f.close()
        # Merge vectors
        new_vectors = np.array(new_vectors)
        vectors = np.array(vectors)
        if DEBUG:
            print("Old vectors: ", vectors.shape)
            print("New vectors: ", new_vectors.shape)
        vectors = np.vstack((vectors, new_vectors))
        if DEBUG:
            print("Combined vectors: ", vectors.shape)
            print("Vocab: ", len(vocab))
        save()
        return True

def vocabularize(_tokens, _report_unks=False, _case_sensitive=False, _embedding=None):
    """
        Given a list of strings (list of tokens), this returns a list of integers (vectorspace ids)
        based on whatever embedding is called in the function, or is the SELECTED EMBEDDING
        :param _tokens: The sentence you want vocabbed. Tokenized (list of str)
        :param _report_unks: Whether or not to return the list of out of vocab tokens
        :param _case_sensitive: Whether or not return to lowercase everything.
        :param _embedding: Which embeddings do you want to use in this process.
        :return: Numpy tensor of n, [OPTIONAL] List(str) of tokens out of vocabulary.
    """
    __check_prepared__(_embedding=_embedding)
    # print(_tokens)
    op = []
    unks = []
    for token in _tokens:
        # Small cap everything
        token = token.lower() if not _case_sensitive else token
        try:
            try:
                token_id = vocab[token]
            except KeyError:
                '''It doesn't exist, give it the _unk_ token, add log it to unks.'''
                if _report_unks:
                    unks.append(token)
                token_id = vocab['_unk_']
                # print('vocab[_unk_]',vocab['_unk_'])
            finally:
                op += [token_id]
        except:
            "This here is to prevent some unknown mishaps, which stops a real long process, and sends me hate mail."
            print(traceback.print_exc())
            print(token)
    return (np.asarray(op), unks) if _report_unks else np.asarray(op)


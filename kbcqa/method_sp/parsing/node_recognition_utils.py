

def set_question_node(ungrounded_nodes):
    '''recogniatize question node'''
    #1. overlap
    is_has = False
    for ungrounded_node in ungrounded_nodes:
        if ungrounded_node.node_type == 'class' and is_question_node_by_words(ungrounded_node.friendly_name):
            ungrounded_node.question_node = 1
            is_has = True
            break
    if not is_has:
        class_nodes_list = []
        for ungrounded_node in ungrounded_nodes:
            if ungrounded_node.node_type == 'class':
                class_nodes_list.append(ungrounded_node)
        if len(class_nodes_list) == 1:
            for class_node in class_nodes_list:
                class_node.question_node = 1
        elif len(class_nodes_list) > 1: # start_position is smaller, is question node
            min_start_position = 100000
            for class_node in class_nodes_list:
                if min_start_position > class_node.start_position:
                    min_start_position = class_node.start_position
            for ungrounded_node in ungrounded_nodes:
                if ungrounded_node.start_position == min_start_position:
                    ungrounded_node.question_node = 1
                    break
    return ungrounded_nodes


wh_words_set = {"what", "which", "whom", "who", "when", "where", "why", "how", "how many", "how large", "how big"}


def is_question_node_by_words(node_mention):
    '''interaction between wh-word and node mention, then the node is question node'''
    result = False
    if node_mention is None: return result
    for word in node_mention.split(' '):
        if word.lower() in wh_words_set:
            result = True
    return result


def is_wh_question_node(node):
    '''node's id is wh-word'''
    result = False
    if node is None: return result
    if node.id in wh_words_set:
        result = True
    return result


def is_equal_wh_word(node_mention):
    '''equal between wh-word and node mention'''
    result = False
    if node_mention is None: return result
    if node_mention.lower() in wh_words_set:
        result = True
    return result


def merge_ner_sequence(ner_sequence):
    '''
    merge ner type, input ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'DATE', 'DATE', 'DATE', 'DATE', 'DATE', 'O']
    return startIndex\tendIndex -> nerType
    '''
    ner_dict = dict()
    ner_type = None
    start_index = 0
    end_index = 0
    is_meet = False
    for index in range(len(ner_sequence)):
        current_tag = ner_sequence[index]
        if '0' != current_tag and 'O' != current_tag:
            if is_meet and ner_type == current_tag: #过程中
                if index != len(ner_sequence) - 1: #没有到达句尾, 过程中
                    continue
            elif is_meet and ner_type != current_tag: # 上一下与下一个不一样了'class', 'relation'
                end_index = index - 1
                # is_meet = False
                ner_dict[str(start_index) + '\t' + str(end_index)] = ner_type
                # ner_type= None
                #相当于又到了一个新mention的开头
                is_meet = True
                start_index = index
                ner_type = current_tag
            else:
                is_meet = True #到了mention的开头
                start_index = index
                ner_type = current_tag

            if index == len(ner_sequence) - 1: # 到达了句尾, 要把当前的mention，追加到ner_dict中
                end_index = index
                ner_dict[str(start_index) + '\t' + str(end_index)] = ner_type

        else:
            if is_meet: #到了mention尾部
                end_index = index - 1
                is_meet = False
                ner_dict[str(start_index) + '\t' + str(end_index)] = ner_type
                ner_type = None
            else:
                continue

    return ner_dict


def char_index_to_token_index(question, start_char, end_char):

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for i, char in enumerate(question):
        if is_whitespace(char):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(char)
            else:
                doc_tokens[-1] += char
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)
    # print('#doc_tokens:\t', doc_tokens)
    # print('#char_to_word_offset:\t', char_to_word_offset)
    start_token_index = char_to_word_offset[start_char]
    if end_char >= len(char_to_word_offset):
        end_token_index = char_to_word_offset[-1]
    else:
        end_token_index = char_to_word_offset[end_char]
    return start_token_index, end_token_index


def get_friendly_name(tokens, sequence_start, sequence_end):
    '''friendly_name = ' '.join([token.value for token in tokens if sequence_end >= token.index >= sequence_start])'''
    friendly_name = ' '.join([token.value for token in tokens if sequence_end >= token.index >= sequence_start])
    # sequence = 'the united states of america'
    # if friendly_name.startswith('the '):
    #     friendly_name = friendly_name[4:]
    return friendly_name.strip()


def get_literal_classifier(friendly_name, is_sutime=False):
    '''type.int, type.float, type.datetime'''
    if is_sutime:
        return 'type.datetime'
    if '.' in friendly_name or '+' in friendly_name:
        return 'type.float'
    else:
        return 'type.int'


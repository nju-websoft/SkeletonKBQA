
def get_old_mention(new_mention):
    '''
    old_mention = get_old_mention('Yes we can !')
    All honor 's wounds are self-inflicted		The devil is God 's ape !	's 前者合并
    if ?后缀, 则删除
    El Gouna Beverage Co. Sakara Gold beer is produced	El Gouna Beverage Co . Sakara Gold	与前者合并
    Forgive your enemies, but never forget their names		Forgive your enemies , but never forget their names .	有标点符号的话，都与前者合并
    King : A Filmed Record ... Montgomery to Memphis	King: A Film Record...Montgomery to Memphis
    Columbia Lions men 's basketball team
    Saami , Lule Language
    The future is ... black .		The future is... black.
    The Climb?	?结尾的, 就删掉
    Canzone , S. 162 no. 2		"How did the composer of \"Canzone, S. 162 no. 2\" earn a living?",
    William DeWitt , Jr.	William DeWitt, Jr. is
    Christmas ( 2011 )		Christmas (2011)
    Yes we can !	Yes we can!
    :param new_mention:
    :return: old mention
    '''
    tokens = new_mention.replace('?', '').split(' ')
    old_mention_list = []
    for i, token in enumerate(tokens):
        if token in ['\'s', '.', ',', '...', '!'] and i > 0:
            old_mention_list.pop()
            old_mention_list.append(tokens[i-1]+token)
        else:
            old_mention_list.append(token)
    old_mention = ' '.join(old_mention_list)
    return old_mention


def add_dict_number(entity_pro_sum, entity_pro_partial):
    '''
    sum all four lexicons
    :param entity_pro_sum:
    :param entity_pro_partial:
    :return: entity_pro_sum, sum dict
    '''
    for entity in entity_pro_partial:
        if entity in entity_pro_sum:
            entity_pro_sum[entity] = entity_pro_sum[entity] + entity_pro_partial[entity]
        else:
            entity_pro_sum[entity] = entity_pro_partial[entity]
    return entity_pro_sum


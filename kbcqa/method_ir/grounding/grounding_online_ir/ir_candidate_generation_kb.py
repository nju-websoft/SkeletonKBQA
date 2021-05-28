import os
import pickle

from method_ir.grounding.grounding_online_ir.candidate_generation_args import oracle_canidates_file

from datasets_interface.virtuoso_interface.dbpedia_sparql_odbc import SparqlQueryODBC
# from datasets_interface.virtuoso_interface.freebase_sparql_odbc import SparqlQueryODBC
# from datasets_interface.virtuoso_interface.dbpedia_sparql_html import SparqlQueryHTML
# from datasets_interface.virtuoso_interface.freebase_sparql_html import SparqlQueryHTML

# sqlodbc = SparqlQueryHTML()
sqlodbc = SparqlQueryODBC()


"""1_0"""


def get_1_0_path_by_entities(e1, e2, bi_direction):
    '''+. -'''
    file_name = e1 + "_" + e2
    plus_topfile = oracle_canidates_file + '+/'
    if not os.path.exists(plus_topfile):
        os.makedirs(plus_topfile)
    if file_name in os.listdir(plus_topfile):
        plus_p1 = pickle.load(open(plus_topfile + file_name, 'rb'))
    else:
        plus_p1 = sqlodbc.get_p1_by_e1_e2(e1=e1, e2=e2, label='+')
        pickle.dump(plus_p1, open(plus_topfile + file_name, 'wb'))
    if not bi_direction:
        return plus_p1

    minus_topfile = oracle_canidates_file + '-/'
    if not os.path.exists(minus_topfile):
        os.makedirs(minus_topfile)
    if file_name in os.listdir(minus_topfile):
        minus_p2 = pickle.load(open(minus_topfile + file_name, 'rb'))
    else:
        minus_p2 = sqlodbc.get_p1_by_e1_e2(e1=e1, e2=e2, label='-')
        pickle.dump(minus_p2, open(minus_topfile + file_name, 'wb'))
    return plus_p1, minus_p2


"""1_1"""


def get_p_o_by_entity(entity):
    '''+'''
    topfile = oracle_canidates_file+'+/'
    if not os.path.exists(topfile):
        os.makedirs(topfile)
    if entity in os.listdir(topfile):
        p_o_set, o_set, p_set = pickle.load(open(topfile+entity,'rb'))
    else:
        p_o_set, o_set, p_set = sqlodbc.get_p_o(entity)
        # p_o_set, o_set, p_set = sqlodbc.get_p_o_literal(entity)
        pickle.dump([p_o_set, o_set, p_set], open(topfile + entity, 'wb'))
    return p_o_set, o_set, p_set


def get_s_p_by_entity(entity):
    '''-'''
    topfile = oracle_canidates_file + '-/'
    if not os.path.exists(topfile):
        os.makedirs(topfile)
    if entity in os.listdir(topfile):
        s_p_set, s_set, p_set = pickle.load(open(topfile + entity, 'rb'))
    else:
        s_p_set, s_set, p_set = sqlodbc.get_s_p(entity)
        pickle.dump([s_p_set, s_set, p_set], open(topfile + entity, 'wb'))
    return s_p_set, s_set, p_set


def get_s_p_by_literal(literal_value):
    '''-'''
    topfile = oracle_canidates_file + '-/'
    if not os.path.exists(topfile):
        os.makedirs(topfile)
    if literal_value in os.listdir(topfile):
        s_p_set, s_set, p_set = pickle.load(open(topfile + literal_value, 'rb'))
    else:
        s_p_set, s_set, p_set = sqlodbc.get_s_p_literal_none(literal_value)
        pickle.dump([s_p_set, s_set, p_set], open(topfile + literal_value, 'wb'))
    return s_p_set, s_set, p_set


"""2-1"""


def get_hop2_1_path_by_entity(entity, bi_direction=False):
    '''++, +-, -+, --'''
    file_name = entity
    plus_plus_topfile = oracle_canidates_file + '++/'
    if not os.path.exists(plus_plus_topfile):
        os.makedirs(plus_plus_topfile)
    if file_name in os.listdir(plus_plus_topfile):
        plus_p1_plus_p2 = pickle.load(open(plus_plus_topfile + entity, 'rb'))
    else:
        plus_p1_plus_p2 = sqlodbc.get_p1_p2_by_entity(entity,'++')
        pickle.dump(plus_p1_plus_p2, open(plus_plus_topfile + file_name, 'wb'))
    if not bi_direction:
        return plus_p1_plus_p2

    plus_minus_topfile = oracle_canidates_file + '+-/'
    if not os.path.exists(plus_minus_topfile):
        os.makedirs(plus_minus_topfile)
    if entity in os.listdir(plus_minus_topfile):
        plus_p1_minus_p2 = pickle.load(open(plus_minus_topfile + entity, 'rb'))
    else:
        plus_p1_minus_p2 = sqlodbc.get_p1_p2_by_entity(entity,'+-')
        pickle.dump(plus_p1_minus_p2, open(plus_minus_topfile + entity, 'wb'))
    minus_plus_topfile = oracle_canidates_file + '-+/'
    if not os.path.exists(minus_plus_topfile):
        os.makedirs(minus_plus_topfile)
    if entity in os.listdir(minus_plus_topfile):
        minus_p1_plus_p2 = pickle.load(open(minus_plus_topfile + entity, 'rb'))
    else:
        minus_p1_plus_p2 = sqlodbc.get_p1_p2_by_entity(entity, '-+')
        pickle.dump(minus_p1_plus_p2, open(minus_plus_topfile + entity, 'wb'))
    minus_minus_topfile = oracle_canidates_file + '--/'
    if not os.path.exists(minus_minus_topfile):
        os.makedirs(minus_minus_topfile)
    if entity in os.listdir(minus_minus_topfile):
        minus_p1_minus_p2 = pickle.load(open(minus_minus_topfile + entity, 'rb'))
    else:
        minus_p1_minus_p2 = sqlodbc.get_p1_p2_by_entity(entity, '--')
        pickle.dump(minus_p1_minus_p2, open(minus_minus_topfile + entity, 'wb'))
    return plus_p1_plus_p2, plus_p1_minus_p2, minus_p1_plus_p2, minus_p1_minus_p2


def get_hop2_1_path_by_literal(literal_value, bi_direction=False):
    '''-+, --'''
    minus_plus_topfile = oracle_canidates_file + '-+/'
    if not os.path.exists(minus_plus_topfile):
        os.makedirs(minus_plus_topfile)
    if literal_value in os.listdir(minus_plus_topfile):
        minus_p1_plus_p2 = pickle.load(open(minus_plus_topfile + literal_value, 'rb'))
    else:
        minus_p1_plus_p2 = sqlodbc.get_p1_p2_by_literal(literal_value, '-+')
        pickle.dump(minus_p1_plus_p2, open(minus_plus_topfile + literal_value, 'wb'))
    if bi_direction:
        return minus_p1_plus_p2

    minus_minus_topfile = oracle_canidates_file + '--/'
    if not os.path.exists(minus_minus_topfile):
        os.makedirs(minus_minus_topfile)
    if literal_value in os.listdir(minus_minus_topfile):
        minus_p1_minus_p2 = pickle.load(open(minus_minus_topfile + literal_value, 'rb'))
    else:
        minus_p1_minus_p2 = sqlodbc.get_p1_p2_by_literal(literal_value, '--')
        pickle.dump(minus_p1_minus_p2, open(minus_minus_topfile + literal_value, 'wb'))
    return minus_p1_plus_p2, minus_p1_minus_p2


"""2-2"""


def get_hop2_2_star_by_entites(e1, e2, bi_direction=False):
    '''++. +-. -+, --'''
    file_name = e1+"_"+e2
    plus_plus_topfile = oracle_canidates_file + '++/'
    if not os.path.exists(plus_plus_topfile):
        os.makedirs(plus_plus_topfile)
    if file_name in os.listdir(plus_plus_topfile):
        plus_p1_plus_p2 = pickle.load(open(plus_plus_topfile + file_name, 'rb'))
    else:
        plus_p1_plus_p2 = sqlodbc.get_p1_p2_by_e1_e2(e1=e1, e2=e2, label='++')
        pickle.dump(plus_p1_plus_p2, open(plus_plus_topfile + file_name, 'wb'))
    if not bi_direction:
        return plus_p1_plus_p2

    plus_minus_topfile = oracle_canidates_file + '+-/'
    if not os.path.exists(plus_minus_topfile):
        os.makedirs(plus_minus_topfile)
    if file_name in os.listdir(plus_minus_topfile):
        plus_p1_minus_p2 = pickle.load(open(plus_minus_topfile + file_name, 'rb'))
    else:
        plus_p1_minus_p2 = sqlodbc.get_p1_p2_by_e1_e2(e1=e1, e2=e2, label='+-')
        pickle.dump(plus_p1_minus_p2, open(plus_minus_topfile + file_name, 'wb'))

    minus_plus_topfile = oracle_canidates_file + '-+/'
    if not os.path.exists(minus_plus_topfile):
        os.makedirs(minus_plus_topfile)
    if file_name in os.listdir(minus_plus_topfile):
        minus_p1_plus_p2 = pickle.load(open(minus_plus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2 = sqlodbc.get_p1_p2_by_e1_e2(e1=e1, e2=e2, label='-+')
        pickle.dump(minus_p1_plus_p2, open(minus_plus_topfile + file_name, 'wb'))

    minus_minus_topfile = oracle_canidates_file + '--/'
    if not os.path.exists(minus_minus_topfile):
        os.makedirs(minus_minus_topfile)
    if file_name in os.listdir(minus_minus_topfile):
        minus_p1_minus_p2 = pickle.load(open(minus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2 = sqlodbc.get_p1_p2_by_e1_e2(e1=e1, e2=e2, label='--')
        pickle.dump(minus_p1_minus_p2, open(minus_minus_topfile + file_name, 'wb'))

    return plus_p1_plus_p2, plus_p1_minus_p2, minus_p1_plus_p2, minus_p1_minus_p2


def get_hop2_2_star_by_literals(l1, l2):
    '''--'''
    file_name = l1+"_"+l2
    minus_minus_topfile = oracle_canidates_file + '--/'
    if file_name in os.listdir(minus_minus_topfile):
        minus_p1_minus_p2 = pickle.load(open(minus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2 = sqlodbc.get_p1_p2_by_e1_e2(e1=l1, e2=l2, label='--')
        pickle.dump(minus_p1_minus_p2, open(minus_minus_topfile + file_name, 'wb'))
    return minus_p1_minus_p2


def get_hop2_2_star_by_entity_literal(l1, e2, bi_direction=False):
    '''-+, --'''
    file_name = l1 + "_" + e2
    minus_plus_topfile = oracle_canidates_file + '-+/'
    if file_name in os.listdir(minus_plus_topfile):
        minus_p1_plus_p2 = pickle.load(open(minus_plus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2 = sqlodbc.get_p1_p2_by_e1_e2(e1=l1, e2=e2, label='-+')
        pickle.dump(minus_p1_plus_p2, open(minus_plus_topfile + file_name, 'wb'))
    if not bi_direction:
        return minus_p1_plus_p2
    minus_minus_topfile = oracle_canidates_file + '--/'
    if file_name in os.listdir(minus_minus_topfile):
        minus_p1_minus_p2 = pickle.load(open(minus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2 = sqlodbc.get_p1_p2_by_e1_e2(e1=l1, e2=e2, label='--')
        pickle.dump(minus_p1_minus_p2, open(minus_minus_topfile + file_name, 'wb'))
    return minus_p1_plus_p2, minus_p1_minus_p2


"""3-3"""


def get_hop3_3_path_by_entity(e1, bi_direction=False):
    '''+++, ---, ...'''
    plus_plus_plus_topfile = oracle_canidates_file + '+++/'
    if not os.path.exists(plus_plus_plus_topfile):
        os.makedirs(plus_plus_plus_topfile)
    if e1 in os.listdir(plus_plus_plus_topfile):
        plus_p1_plus_p2_plus_p3 = pickle.load(open(plus_plus_plus_topfile + e1, 'rb'))
    else:
        plus_p1_plus_p2_plus_p3 = sqlodbc.get_p1_p2_p3_by_entity(s=e1, label='+++')
        pickle.dump(plus_p1_plus_p2_plus_p3, open(plus_plus_plus_topfile + e1, 'wb'))
    if not bi_direction:
        return plus_p1_plus_p2_plus_p3

    plus_plus_minus_topfile = oracle_canidates_file + '++-/'
    if not os.path.exists(plus_plus_minus_topfile):
        os.makedirs(plus_plus_minus_topfile)
    if e1 in os.listdir(plus_plus_minus_topfile):
        plus_p1_plus_p2_minus_p3 = pickle.load(open(plus_plus_minus_topfile + e1, 'rb'))
    else:
        plus_p1_plus_p2_minus_p3 = sqlodbc.get_p1_p2_p3_by_entity(s=e1, label='++-')
        pickle.dump(plus_p1_plus_p2_minus_p3, open(plus_plus_minus_topfile + e1, 'wb'))

    plus_minus_plus_topfile = oracle_canidates_file + '+-+/'
    if not os.path.exists(plus_minus_plus_topfile):
        os.makedirs(plus_minus_plus_topfile)
    if e1 in os.listdir(plus_minus_plus_topfile):
        plus_p1_minus_p2_plus_p3 = pickle.load(open(plus_minus_plus_topfile + e1, 'rb'))
    else:
        plus_p1_minus_p2_plus_p3 = sqlodbc.get_p1_p2_p3_by_entity(s=e1, label='+-+')
        pickle.dump(plus_p1_minus_p2_plus_p3, open(plus_minus_plus_topfile + e1, 'wb'))

    plus_minus_minus_topfile = oracle_canidates_file + '+--/'
    if not os.path.exists(plus_minus_minus_topfile):
        os.makedirs(plus_minus_minus_topfile)
    if e1 in os.listdir(plus_minus_minus_topfile):
        plus_p1_minus_p2_minus_p3 = pickle.load(open(plus_minus_minus_topfile + e1, 'rb'))
    else:
        plus_p1_minus_p2_minus_p3 = sqlodbc.get_p1_p2_p3_by_entity(s=e1, label='+--')
        pickle.dump(plus_p1_minus_p2_minus_p3, open(plus_minus_minus_topfile + e1, 'wb'))

    minus_plus_plus_topfile = oracle_canidates_file + '-++/'
    if not os.path.exists(minus_plus_plus_topfile):
        os.makedirs(minus_plus_plus_topfile)
    if e1 in os.listdir(minus_plus_plus_topfile):
        minus_p1_plus_p2_plus_p3 = pickle.load(open(minus_plus_plus_topfile + e1, 'rb'))
    else:
        minus_p1_plus_p2_plus_p3 = sqlodbc.get_p1_p2_p3_by_entity(s=e1, label='-++')
        pickle.dump(minus_p1_plus_p2_plus_p3, open(minus_plus_plus_topfile + e1, 'wb'))

    minus_plus_minus_topfile = oracle_canidates_file + '-+-/'
    if not os.path.exists(minus_plus_minus_topfile):
        os.makedirs(minus_plus_minus_topfile)
    if e1 in os.listdir(minus_plus_minus_topfile):
        minus_p1_plus_p2_minus_p3 = pickle.load(open(minus_plus_minus_topfile + e1, 'rb'))
    else:
        minus_p1_plus_p2_minus_p3 = sqlodbc.get_p1_p2_p3_by_entity(s=e1, label='-+-')
        pickle.dump(minus_p1_plus_p2_minus_p3, open(minus_plus_minus_topfile + e1, 'wb'))

    minus_minus_plus_topfile = oracle_canidates_file + '--+/'
    if not os.path.exists(minus_minus_plus_topfile):
        os.makedirs(minus_minus_plus_topfile)
    if e1 in os.listdir(minus_minus_plus_topfile):
        minus_p1_minus_p2_plus_p3 = pickle.load(open(minus_minus_plus_topfile + e1, 'rb'))
    else:
        minus_p1_minus_p2_plus_p3 = sqlodbc.get_p1_p2_p3_by_entity(s=e1, label='--+')
        pickle.dump(minus_p1_minus_p2_plus_p3, open(minus_minus_plus_topfile + e1, 'wb'))

    minus_minus_minus_topfile = oracle_canidates_file + '---/'
    if not os.path.exists(minus_minus_minus_topfile):
        os.makedirs(minus_minus_minus_topfile)
    if e1 in os.listdir(minus_minus_minus_topfile):
        minus_p1_minus_p2_minus_p3 = pickle.load(open(minus_minus_minus_topfile + e1, 'rb'))
    else:
        minus_p1_minus_p2_minus_p3 = sqlodbc.get_p1_p2_p3_by_entity(s=e1, label='---')
        pickle.dump(minus_p1_minus_p2_minus_p3, open(minus_minus_minus_topfile + e1, 'wb'))

    return plus_p1_plus_p2_plus_p3, plus_p1_plus_p2_minus_p3, plus_p1_minus_p2_plus_p3, plus_p1_minus_p2_minus_p3, \
        minus_p1_plus_p2_plus_p3, minus_p1_plus_p2_minus_p3, minus_p1_minus_p2_plus_p3, minus_p1_minus_p2_minus_p3


def get_hop3_3_path_by_literal(l1, bi_direction=False):
    '''---, ...'''
    minus_plus_plus_topfile = oracle_canidates_file + '-++/'
    if not os.path.exists(minus_plus_plus_topfile):
        os.makedirs(minus_plus_plus_topfile)
    if l1 in os.listdir(minus_plus_plus_topfile):
        minus_p1_plus_p2_plus_p3 = pickle.load(open(minus_plus_plus_topfile + l1, 'rb'))
    else:
        minus_p1_plus_p2_plus_p3 = sqlodbc.get_p1_p2_p3_by_entity(s=l1, label='-++')
        pickle.dump(minus_p1_plus_p2_plus_p3, open(minus_plus_plus_topfile + l1, 'wb'))

    if not bi_direction:
        return minus_p1_plus_p2_plus_p3

    minus_plus_minus_topfile = oracle_canidates_file + '-+-/'
    if not os.path.exists(minus_plus_minus_topfile):
        os.makedirs(minus_plus_minus_topfile)
    if l1 in os.listdir(minus_plus_minus_topfile):
        minus_p1_plus_p2_minus_p3 = pickle.load(open(minus_plus_minus_topfile + l1, 'rb'))
    else:
        minus_p1_plus_p2_minus_p3 = sqlodbc.get_p1_p2_p3_by_entity(s=l1, label='-+-')
        pickle.dump(minus_p1_plus_p2_minus_p3, open(minus_plus_minus_topfile + l1, 'wb'))

    minus_minus_plus_topfile = oracle_canidates_file + '--+/'
    if not os.path.exists(minus_minus_plus_topfile):
        os.makedirs(minus_minus_plus_topfile)
    if l1 in os.listdir(minus_minus_plus_topfile):
        minus_p1_minus_p2_plus_p3 = pickle.load(open(minus_minus_plus_topfile + l1, 'rb'))
    else:
        minus_p1_minus_p2_plus_p3 = sqlodbc.get_p1_p2_p3_by_entity(s=l1, label='--+')
        pickle.dump(minus_p1_minus_p2_plus_p3, open(minus_minus_plus_topfile + l1, 'wb'))

    minus_minus_minus_topfile = oracle_canidates_file + '---/'
    if not os.path.exists(minus_minus_minus_topfile):
        os.makedirs(minus_minus_minus_topfile)
    if l1 in os.listdir(minus_minus_minus_topfile):
        minus_p1_minus_p2_minus_p3 = pickle.load(open(minus_minus_minus_topfile + l1, 'rb'))
    else:
        minus_p1_minus_p2_minus_p3 = sqlodbc.get_p1_p2_p3_by_entity(s=l1, label='---')
        pickle.dump(minus_p1_minus_p2_minus_p3, open(minus_minus_minus_topfile + l1, 'wb'))

    return minus_p1_plus_p2_plus_p3, minus_p1_plus_p2_minus_p3, minus_p1_minus_p2_plus_p3, minus_p1_minus_p2_minus_p3


"""3-1"""


def get_hop3_1_star_by_entities(e1, e2, bi_direction=False, left_or_right_insertNode='left'):
    '''+++'''
    file_name = e1 + "_" + e2
    plus_plus_plus_topfile = oracle_canidates_file + '+++/'
    if not os.path.exists(plus_plus_plus_topfile):
        os.makedirs(plus_plus_plus_topfile)
    if file_name in os.listdir(plus_plus_plus_topfile):
        plus_p1_plus_p2_plus_p3 = pickle.load(open(plus_plus_plus_topfile + file_name, 'rb'))
    else:
        plus_p1_plus_p2_plus_p3 = sqlodbc.get_p1_p2_p3_by_e1_e2(e1=e1, e2=e2, label='+++', left_or_right_insertNode=left_or_right_insertNode)
        pickle.dump(plus_p1_plus_p2_plus_p3, open(plus_plus_plus_topfile + file_name, 'wb'))
    if not bi_direction:
        return plus_p1_plus_p2_plus_p3

    plus_plus_minus_topfile = oracle_canidates_file + '++-/'
    if not os.path.exists(plus_plus_minus_topfile):
        os.makedirs(plus_plus_minus_topfile)
    if file_name in os.listdir(plus_plus_minus_topfile):
        plus_p1_plus_p2_minus_p3 = pickle.load(open(plus_plus_minus_topfile + file_name, 'rb'))
    else:
        plus_p1_plus_p2_minus_p3 = sqlodbc.get_p1_p2_p3_by_e1_e2(e1=e1, e2=e2, label='++-', left_or_right_insertNode=left_or_right_insertNode)
        pickle.dump(plus_p1_plus_p2_minus_p3, open(plus_plus_minus_topfile + file_name, 'wb'))

    plus_minus_plus_topfile = oracle_canidates_file + '+-+/'
    if not os.path.exists(plus_minus_plus_topfile):
        os.makedirs(plus_minus_plus_topfile)
    if file_name in os.listdir(plus_minus_plus_topfile):
        plus_p1_minus_p2_plus_p3 = pickle.load(open(plus_minus_plus_topfile + file_name, 'rb'))
    else:
        plus_p1_minus_p2_plus_p3 = sqlodbc.get_p1_p2_p3_by_e1_e2(e1=e1, e2=e2, label='+-+', left_or_right_insertNode=left_or_right_insertNode)
        pickle.dump(plus_p1_minus_p2_plus_p3, open(plus_minus_plus_topfile + file_name, 'wb'))

    plus_minus_minus_topfile = oracle_canidates_file + '+--/'
    if not os.path.exists(plus_minus_minus_topfile):
        os.makedirs(plus_minus_minus_topfile)
    if file_name in os.listdir(plus_minus_minus_topfile):
        plus_p1_minus_p2_minus_p3 = pickle.load(open(plus_minus_minus_topfile + file_name, 'rb'))
    else:
        plus_p1_minus_p2_minus_p3 = sqlodbc.get_p1_p2_p3_by_e1_e2(e1=e1, e2=e2, label='+--', left_or_right_insertNode=left_or_right_insertNode)
        pickle.dump(plus_p1_minus_p2_minus_p3, open(plus_minus_minus_topfile + file_name, 'wb'))

    minus_plus_plus_topfile = oracle_canidates_file + '-++/'
    if not os.path.exists(minus_plus_plus_topfile):
        os.makedirs(minus_plus_plus_topfile)
    if file_name in os.listdir(minus_plus_plus_topfile):
        minus_p1_plus_p2_plus_p3 = pickle.load(open(minus_plus_plus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_plus_p3 = sqlodbc.get_p1_p2_p3_by_e1_e2(e1=e1, e2=e2, label='-++', left_or_right_insertNode=left_or_right_insertNode)
        pickle.dump(minus_p1_plus_p2_plus_p3, open(minus_plus_plus_topfile + file_name, 'wb'))

    minus_plus_minus_topfile = oracle_canidates_file + '-+-/'
    if not os.path.exists(minus_plus_minus_topfile):
        os.makedirs(minus_plus_minus_topfile)
    if file_name in os.listdir(minus_plus_minus_topfile):
        minus_p1_plus_p2_minus_p3 = pickle.load(open(minus_plus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_minus_p3 = sqlodbc.get_p1_p2_p3_by_e1_e2(e1=e1, e2=e2, label='-+-', left_or_right_insertNode=left_or_right_insertNode)
        pickle.dump(minus_p1_plus_p2_minus_p3, open(minus_plus_minus_topfile + file_name, 'wb'))

    minus_minus_plus_topfile = oracle_canidates_file + '--+/'
    if not os.path.exists(minus_minus_plus_topfile):
        os.makedirs(minus_minus_plus_topfile)
    if file_name in os.listdir(minus_minus_plus_topfile):
        minus_p1_minus_p2_plus_p3 = pickle.load(open(minus_minus_plus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_plus_p3 = sqlodbc.get_p1_p2_p3_by_e1_e2(e1=e1, e2=e2, label='--+', left_or_right_insertNode=left_or_right_insertNode)
        pickle.dump(minus_p1_minus_p2_plus_p3, open(minus_minus_plus_topfile + file_name, 'wb'))

    minus_minus_minus_topfile = oracle_canidates_file + '---/'
    if not os.path.exists(minus_minus_minus_topfile):
        os.makedirs(minus_minus_minus_topfile)
    if file_name in os.listdir(plus_minus_plus_topfile):
        minus_p1_minus_p2_minus_p3 = pickle.load(open(minus_minus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_minus_p3 = sqlodbc.get_p1_p2_p3_by_e1_e2(e1=e1, e2=e2, label='---', left_or_right_insertNode=left_or_right_insertNode)
        pickle.dump(minus_p1_minus_p2_minus_p3, open(minus_minus_minus_topfile + file_name, 'wb'))

    return plus_p1_plus_p2_plus_p3, plus_p1_plus_p2_minus_p3, plus_p1_minus_p2_plus_p3, plus_p1_minus_p2_minus_p3,\
            minus_p1_plus_p2_plus_p3, minus_p1_plus_p2_minus_p3, minus_p1_minus_p2_plus_p3, minus_p1_minus_p2_minus_p3


def get_hop3_1_star_by_literals(l1, l2, bi_direction=False, left_or_right_insertNode='left'):
    '''-+-, ---'''
    file_name = l1 + "_" + l2
    minus_plus_minus_topfile = oracle_canidates_file + '-+-/'
    if not os.path.exists(minus_plus_minus_topfile):
        os.makedirs(minus_plus_minus_topfile)
    if file_name in os.listdir(minus_plus_minus_topfile):
        minus_p1_plus_p2_minus_p3 = pickle.load(open(minus_plus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_minus_p3 = sqlodbc.get_p1_p2_p3_by_e1_e2(e1=l1, e2=l2, label='-+-', left_or_right_insertNode=left_or_right_insertNode)
        pickle.dump(minus_p1_plus_p2_minus_p3, open(minus_plus_minus_topfile + file_name, 'wb'))
    if not bi_direction:
        return minus_p1_plus_p2_minus_p3

    minus_minus_minus_topfile = oracle_canidates_file + '---/'
    if not os.path.exists(minus_minus_minus_topfile):
        os.makedirs(minus_minus_minus_topfile)
    if file_name in os.listdir(minus_minus_minus_topfile):
        minus_p1_minus_p2_minus_p3 = pickle.load(open(minus_minus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_minus_p3 = sqlodbc.get_p1_p2_p3_by_e1_e2(e1=l1, e2=l2, label='---', left_or_right_insertNode=left_or_right_insertNode)
        pickle.dump(minus_p1_minus_p2_minus_p3, open(minus_minus_minus_topfile + file_name, 'wb'))
    return minus_p1_plus_p2_minus_p3, minus_p1_minus_p2_minus_p3


def get_hop3_1_star_by_entity_literal(e1, l2, bi_direction=False, left_or_right_insertNode='left'):
    '''++-'''
    file_name = e1 + "_" + l2
    plus_plus_minus_topfile = oracle_canidates_file + '++-/'
    if not os.path.exists(plus_plus_minus_topfile):
        os.makedirs(plus_plus_minus_topfile)
    if file_name in os.listdir(plus_plus_minus_topfile):
        plus_p1_plus_p2_minus_p3 = pickle.load(open(plus_plus_minus_topfile + file_name, 'rb'))
    else:
        plus_p1_plus_p2_minus_p3 = sqlodbc.get_p1_p2_p3_by_e1_e2(e1=e1, e2=l2, label='++-', left_or_right_insertNode=left_or_right_insertNode)
        pickle.dump(plus_p1_plus_p2_minus_p3, open(plus_plus_minus_topfile + file_name, 'wb'))

    if not bi_direction:
        return plus_p1_plus_p2_minus_p3

    plus_minus_minus_topfile = oracle_canidates_file + '+--/'
    if not os.path.exists(plus_minus_minus_topfile):
        os.makedirs(plus_minus_minus_topfile)
    if file_name in os.listdir(plus_minus_minus_topfile):
        plus_p1_minus_p2_minus_p3 = pickle.load(open(plus_minus_minus_topfile + file_name, 'rb'))
    else:
        plus_p1_minus_p2_minus_p3 = sqlodbc.get_p1_p2_p3_by_e1_e2(e1=e1, e2=l2, label='+--', left_or_right_insertNode=left_or_right_insertNode)
        pickle.dump(plus_p1_minus_p2_minus_p3, open(plus_minus_minus_topfile + file_name, 'wb'))

    minus_plus_minus_topfile = oracle_canidates_file + '-+-/'
    if not os.path.exists(minus_plus_minus_topfile):
        os.makedirs(minus_plus_minus_topfile)
    if file_name in os.listdir(minus_plus_minus_topfile):
        minus_p1_plus_p2_minus_p3 = pickle.load(open(minus_plus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_minus_p3 = sqlodbc.get_p1_p2_p3_by_e1_e2(e1=e1, e2=l2, label='-+-', left_or_right_insertNode=left_or_right_insertNode)
        pickle.dump(minus_p1_plus_p2_minus_p3, open(minus_plus_minus_topfile + file_name, 'wb'))

    minus_minus_minus_topfile = oracle_canidates_file + '---/'
    if not os.path.exists(minus_minus_minus_topfile):
        os.makedirs(minus_minus_minus_topfile)
    if file_name in os.listdir(minus_minus_minus_topfile):
        minus_p1_minus_p2_minus_p3 = pickle.load(open(minus_minus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_minus_p3 = sqlodbc.get_p1_p2_p3_by_e1_e2(e1=e1, e2=l2, label='---', left_or_right_insertNode=left_or_right_insertNode)
        pickle.dump(minus_p1_minus_p2_minus_p3, open(minus_minus_minus_topfile + file_name, 'wb'))

    return plus_p1_plus_p2_minus_p3, plus_p1_minus_p2_minus_p3, minus_p1_plus_p2_minus_p3, minus_p1_minus_p2_minus_p3


"""3-2"""


def get_hop3_2_star_by_entities(e1, e2, bi_direction=False):
    return get_hop3_1_star_by_entities(e1=e1, e2=e2, bi_direction=bi_direction, left_or_right_insertNode='right')


def get_hop3_2_star_by_literals(l1, l2, bi_direction=False):
    return get_hop3_1_star_by_literals(l1=l1, l2=l2, bi_direction=bi_direction, left_or_right_insertNode='right')


def get_hop3_2_star_by_entity_literal(e1, l2, bi_direction=False):
    return get_hop3_1_star_by_entity_literal(e1=e1, l2=l2, bi_direction=bi_direction, left_or_right_insertNode='right')


"""4-1"""


def get_hop4_1_star_by_entities(e1, e2, bi_direction=False):
    '''+++'''
    file_name = e1 + "_" + e2
    plus_plus_plus_plus_topfile = oracle_canidates_file + '++++/'
    if not os.path.exists(plus_plus_plus_plus_topfile):
        os.makedirs(plus_plus_plus_plus_topfile)
    if file_name in os.listdir(plus_plus_plus_plus_topfile):
        plus_p1_plus_p2_plus_p3_plus_p4 = pickle.load(open(plus_plus_plus_plus_topfile + file_name, 'rb'))
    else:
        plus_p1_plus_p2_plus_p3_plus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=e2, label='++++')
        pickle.dump(plus_p1_plus_p2_plus_p3_plus_p4, open(plus_plus_plus_plus_topfile + file_name, 'wb'))
    if not bi_direction:
        return plus_p1_plus_p2_plus_p3_plus_p4

    plus_plus_plus_minus_topfile = oracle_canidates_file + '+++-/'
    if not os.path.exists(plus_plus_plus_minus_topfile):
        os.makedirs(plus_plus_plus_minus_topfile)
    if file_name in os.listdir(plus_plus_plus_minus_topfile):
        plus_p1_plus_p2_plus_p3_minus_p4 = pickle.load(open(plus_plus_plus_minus_topfile + file_name, 'rb'))
    else:
        plus_p1_plus_p2_plus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=e2, label='+++-')
        pickle.dump(plus_p1_plus_p2_plus_p3_minus_p4, open(plus_plus_plus_minus_topfile + file_name, 'wb'))

    plus_plus_minus_plus_topfile = oracle_canidates_file + '++-+/'
    if not os.path.exists(plus_plus_minus_plus_topfile):
        os.makedirs(plus_plus_minus_plus_topfile)
    if file_name in os.listdir(plus_plus_minus_plus_topfile):
        plus_p1_plus_p2_minus_p3_plus_p4 = pickle.load(open(plus_plus_minus_plus_topfile + file_name, 'rb'))
    else:
        plus_p1_plus_p2_minus_p3_plus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=e2, label='++-+')
        pickle.dump(plus_p1_plus_p2_minus_p3_plus_p4, open(plus_plus_minus_plus_topfile + file_name, 'wb'))

    plus_plus_minus_minus_topfile = oracle_canidates_file + '++--/'
    if not os.path.exists(plus_plus_minus_minus_topfile):
        os.makedirs(plus_plus_minus_minus_topfile)
    if file_name in os.listdir(plus_plus_minus_minus_topfile):
        plus_p1_plus_p2_minus_p3_minus_p4 = pickle.load(open(plus_plus_minus_minus_topfile + file_name, 'rb'))
    else:
        plus_p1_plus_p2_minus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=e2, label='++--')
        pickle.dump(plus_p1_plus_p2_minus_p3_minus_p4, open(plus_plus_minus_minus_topfile + file_name, 'wb'))

    plus_minus_plus_plus_topfile = oracle_canidates_file + '+-++/'
    if not os.path.exists(plus_minus_plus_plus_topfile):
        os.makedirs(plus_minus_plus_plus_topfile)
    if file_name in os.listdir(plus_minus_plus_plus_topfile):
        plus_p1_minus_p2_plus_p3_plus_p4 = pickle.load(open(plus_minus_plus_plus_topfile + file_name, 'rb'))
    else:
        plus_p1_minus_p2_plus_p3_plus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=e2, label='+-++')
        pickle.dump(plus_p1_minus_p2_plus_p3_plus_p4, open(plus_minus_plus_plus_topfile + file_name, 'wb'))

    plus_minus_plus_minus_topfile = oracle_canidates_file + '+-+-/'
    if not os.path.exists(plus_minus_plus_minus_topfile):
        os.makedirs(plus_minus_plus_minus_topfile)
    if file_name in os.listdir(plus_minus_plus_minus_topfile):
        plus_p1_minus_p2_plus_p3_minus_p4 = pickle.load(open(plus_minus_plus_minus_topfile + file_name, 'rb'))
    else:
        plus_p1_minus_p2_plus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=e2, label='+-+-')
        pickle.dump(plus_p1_minus_p2_plus_p3_minus_p4, open(plus_minus_plus_minus_topfile + file_name, 'wb'))

    plus_minus_minus_plus_topfile = oracle_canidates_file + '+--+/'
    if not os.path.exists(plus_minus_minus_plus_topfile):
        os.makedirs(plus_minus_minus_plus_topfile)
    if file_name in os.listdir(plus_minus_minus_plus_topfile):
        plus_p1_minus_p2_minus_p3_plus_p4 = pickle.load(open(plus_minus_minus_plus_topfile + file_name, 'rb'))
    else:
        plus_p1_minus_p2_minus_p3_plus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=e2, label='+--+')
        pickle.dump(plus_p1_minus_p2_minus_p3_plus_p4, open(plus_minus_minus_plus_topfile + file_name, 'wb'))

    plus_minus_minus_minus_topfile = oracle_canidates_file + '+---/'
    if not os.path.exists(plus_minus_minus_minus_topfile):
        os.makedirs(plus_minus_minus_minus_topfile)
    if file_name in os.listdir(plus_minus_minus_minus_topfile):
        plus_p1_minus_p2_minus_p3_minus_p4 = pickle.load(open(plus_minus_minus_minus_topfile + file_name, 'rb'))
    else:
        plus_p1_minus_p2_minus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=e2, label='+---')
        pickle.dump(plus_p1_minus_p2_minus_p3_minus_p4, open(plus_minus_minus_minus_topfile + file_name, 'wb'))

    minus_plus_plus_plus_topfile = oracle_canidates_file + '-+++/'
    if not os.path.exists(minus_plus_plus_plus_topfile):
        os.makedirs(minus_plus_plus_plus_topfile)
    if file_name in os.listdir(minus_plus_plus_plus_topfile):
        minus_p1_plus_p2_plus_p3_plus_p4 = pickle.load(open(minus_plus_plus_plus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_plus_p3_plus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=e2, label='-+++')
        pickle.dump(minus_p1_plus_p2_plus_p3_plus_p4, open(minus_plus_plus_plus_topfile + file_name, 'wb'))

    minus_plus_plus_minus_topfile = oracle_canidates_file + '-++-/'
    if not os.path.exists(minus_plus_plus_minus_topfile):
        os.makedirs(minus_plus_plus_minus_topfile)
    if file_name in os.listdir(minus_plus_plus_minus_topfile):
        minus_p1_plus_p2_plus_p3_minus_p4 = pickle.load(open(minus_plus_plus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_plus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=e2, label='-++-')
        pickle.dump(minus_p1_plus_p2_plus_p3_minus_p4, open(minus_plus_plus_minus_topfile + file_name, 'wb'))

    minus_plus_minus_plus_topfile = oracle_canidates_file + '-+-+/'
    if not os.path.exists(minus_plus_minus_plus_topfile):
        os.makedirs(minus_plus_minus_plus_topfile)
    if file_name in os.listdir(minus_plus_minus_plus_topfile):
        minus_p1_plus_p2_minus_p3_plus_p4 = pickle.load(open(minus_plus_minus_plus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_minus_p3_plus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=e2, label='-+-+')
        pickle.dump(minus_p1_plus_p2_minus_p3_plus_p4, open(minus_plus_minus_plus_topfile + file_name, 'wb'))

    minus_plus_minus_minus_topfile = oracle_canidates_file + '-+--/'
    if not os.path.exists(minus_plus_minus_minus_topfile):
        os.makedirs(minus_plus_minus_minus_topfile)
    if file_name in os.listdir(minus_plus_minus_minus_topfile):
        minus_p1_plus_p2_minus_p3_minus_p4 = pickle.load(open(minus_plus_minus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_minus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=e2, label='-+--')
        pickle.dump(minus_p1_plus_p2_minus_p3_minus_p4, open(minus_plus_minus_minus_topfile + file_name, 'wb'))

    minus_minus_plus_plus_topfile = oracle_canidates_file + '--++/'
    if not os.path.exists(minus_minus_plus_plus_topfile):
        os.makedirs(minus_minus_plus_plus_topfile)
    if file_name in os.listdir(minus_minus_plus_plus_topfile):
        minus_p1_minus_p2_plus_p3_plus_p4 = pickle.load(open(minus_minus_plus_plus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_plus_p3_plus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=e2, label='--++')
        pickle.dump(minus_p1_minus_p2_plus_p3_plus_p4, open(minus_minus_plus_plus_topfile + file_name, 'wb'))

    minus_minus_plus_minus_topfile = oracle_canidates_file + '--+-/'
    if not os.path.exists(minus_minus_plus_minus_topfile):
        os.makedirs(minus_minus_plus_minus_topfile)
    if file_name in os.listdir(minus_minus_plus_minus_topfile):
        minus_p1_minus_p2_plus_p3_minus_p4 = pickle.load(open(minus_minus_plus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_plus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=e2, label='--+-')
        pickle.dump(minus_p1_minus_p2_plus_p3_minus_p4, open(minus_minus_plus_minus_topfile + file_name, 'wb'))

    minus_minus_minus_plus_topfile = oracle_canidates_file + '---+/'
    if not os.path.exists(minus_minus_minus_plus_topfile):
        os.makedirs(minus_minus_minus_plus_topfile)
    if file_name in os.listdir(minus_minus_minus_plus_topfile):
        minus_p1_minus_p2_minus_p3_plus_p4 = pickle.load(open(minus_minus_minus_plus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_minus_p3_plus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=e2, label='---+')
        pickle.dump(minus_p1_minus_p2_minus_p3_plus_p4, open(minus_minus_minus_plus_topfile + file_name, 'wb'))

    minus_minus_minus_minus_topfile = oracle_canidates_file + '----/'
    if not os.path.exists(minus_minus_minus_minus_topfile):
        os.makedirs(minus_minus_minus_minus_topfile)
    if file_name in os.listdir(minus_minus_minus_minus_topfile):
        minus_p1_minus_p2_minus_p3_minus_p4 = pickle.load(open(minus_minus_minus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_minus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=e2, label='----')
        pickle.dump(minus_p1_minus_p2_minus_p3_minus_p4, open(minus_minus_minus_minus_topfile + file_name, 'wb'))

    return  plus_p1_plus_p2_plus_p3_plus_p4, plus_p1_plus_p2_plus_p3_minus_p4, plus_p1_plus_p2_minus_p3_plus_p4, plus_p1_plus_p2_minus_p3_minus_p4, \
            plus_p1_minus_p2_plus_p3_plus_p4, plus_p1_minus_p2_plus_p3_minus_p4, plus_p1_minus_p2_minus_p3_plus_p4, plus_p1_minus_p2_minus_p3_minus_p4, \
            minus_p1_plus_p2_plus_p3_plus_p4, minus_p1_plus_p2_plus_p3_minus_p4, minus_p1_plus_p2_minus_p3_plus_p4, minus_p1_plus_p2_minus_p3_minus_p4, \
            minus_p1_minus_p2_plus_p3_plus_p4, minus_p1_minus_p2_plus_p3_minus_p4, minus_p1_minus_p2_minus_p3_plus_p4, minus_p1_minus_p2_minus_p3_minus_p4


def get_hop4_1_star_by_literals(l1, l2, bi_direction=False):
    '''++++'''
    file_name = l1 + "_" + l2
    minus_plus_plus_minus_topfile = oracle_canidates_file + '-++-/'
    if not os.path.exists(minus_plus_plus_minus_topfile):
        os.makedirs(minus_plus_plus_minus_topfile)
    if file_name in os.listdir(minus_plus_plus_minus_topfile):
        minus_p1_plus_p2_plus_p3_minus_p4 = pickle.load(open(minus_plus_plus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_plus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=l1, e2=l2, label='-++-')
        pickle.dump(minus_p1_plus_p2_plus_p3_minus_p4, open(minus_plus_plus_minus_topfile + file_name, 'wb'))
    if not bi_direction:
        return minus_p1_plus_p2_plus_p3_minus_p4

    minus_plus_minus_minus_topfile = oracle_canidates_file + '-+--/'
    if not os.path.exists(minus_plus_minus_minus_topfile):
        os.makedirs(minus_plus_minus_minus_topfile)
    if file_name in os.listdir(minus_plus_minus_minus_topfile):
        minus_p1_plus_p2_minus_p3_minus_p4 = pickle.load(open(minus_plus_minus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_minus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=l1, e2=l2, label='-+--')
        pickle.dump(minus_p1_plus_p2_minus_p3_minus_p4, open(minus_plus_minus_minus_topfile + file_name, 'wb'))

    minus_minus_plus_minus_topfile = oracle_canidates_file + '--+-/'
    if not os.path.exists(minus_minus_plus_minus_topfile):
        os.makedirs(minus_minus_plus_minus_topfile)
    if file_name in os.listdir(minus_minus_plus_minus_topfile):
        minus_p1_minus_p2_plus_p3_minus_p4 = pickle.load(open(minus_minus_plus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_plus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=l1, e2=l2, label='--+-')
        pickle.dump(minus_p1_minus_p2_plus_p3_minus_p4, open(minus_minus_plus_minus_topfile + file_name, 'wb'))

    minus_minus_minus_minus_topfile = oracle_canidates_file + '----/'
    if not os.path.exists(minus_minus_minus_minus_topfile):
        os.makedirs(minus_minus_minus_minus_topfile)
    if file_name in os.listdir(minus_minus_minus_minus_topfile):
        minus_p1_minus_p2_minus_p3_minus_p4 = pickle.load(open(minus_minus_minus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_minus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=l1, e2=l2, label='----')
        pickle.dump(minus_p1_minus_p2_minus_p3_minus_p4, open(minus_minus_minus_minus_topfile + file_name, 'wb'))

    return minus_p1_plus_p2_plus_p3_minus_p4, minus_p1_plus_p2_minus_p3_minus_p4, \
           minus_p1_minus_p2_plus_p3_minus_p4, minus_p1_minus_p2_minus_p3_minus_p4


def get_hop4_1_star_by_entity_literal(e1, l2, bi_direction=False):
    '''+++'''
    file_name = e1 + "_" + l2
    plus_plus_plus_minus_topfile = oracle_canidates_file + '+++-/'
    if not os.path.exists(plus_plus_plus_minus_topfile):
        os.makedirs(plus_plus_plus_minus_topfile)
    if file_name in os.listdir(plus_plus_plus_minus_topfile):
        plus_p1_plus_p2_plus_p3_minus_p4 = pickle.load(open(plus_plus_plus_minus_topfile + file_name, 'rb'))
    else:
        plus_p1_plus_p2_plus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=l2, label='+++-')
        pickle.dump(plus_p1_plus_p2_plus_p3_minus_p4, open(plus_plus_plus_minus_topfile + file_name, 'wb'))
    if not bi_direction:
        return plus_p1_plus_p2_plus_p3_minus_p4

    plus_plus_minus_minus_topfile = oracle_canidates_file + '++--/'
    if not os.path.exists(plus_plus_minus_minus_topfile):
        os.makedirs(plus_plus_minus_minus_topfile)
    if file_name in os.listdir(plus_plus_minus_minus_topfile):
        plus_p1_plus_p2_minus_p3_minus_p4 = pickle.load(open(plus_plus_minus_minus_topfile + file_name, 'rb'))
    else:
        plus_p1_plus_p2_minus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=l2, label='++--')
        pickle.dump(plus_p1_plus_p2_minus_p3_minus_p4, open(plus_plus_minus_minus_topfile + file_name, 'wb'))

    plus_minus_plus_minus_topfile = oracle_canidates_file + '+-+-/'
    if not os.path.exists(plus_minus_plus_minus_topfile):
        os.makedirs(plus_minus_plus_minus_topfile)
    if file_name in os.listdir(plus_minus_plus_minus_topfile):
        plus_p1_minus_p2_plus_p3_minus_p4 = pickle.load(open(plus_minus_plus_minus_topfile + file_name, 'rb'))
    else:
        plus_p1_minus_p2_plus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=l2, label='+-+-')
        pickle.dump(plus_p1_minus_p2_plus_p3_minus_p4, open(plus_minus_plus_minus_topfile + file_name, 'wb'))

    plus_minus_minus_minus_topfile = oracle_canidates_file + '+---/'
    if not os.path.exists(plus_minus_minus_minus_topfile):
        os.makedirs(plus_minus_minus_minus_topfile)
    if file_name in os.listdir(plus_minus_minus_minus_topfile):
        plus_p1_minus_p2_minus_p3_minus_p4 = pickle.load(open(plus_minus_minus_minus_topfile + file_name, 'rb'))
    else:
        plus_p1_minus_p2_minus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=l2, label='+---')
        pickle.dump(plus_p1_minus_p2_minus_p3_minus_p4, open(plus_minus_minus_minus_topfile + file_name, 'wb'))

    minus_plus_plus_minus_topfile = oracle_canidates_file + '-++-/'
    if not os.path.exists(minus_plus_plus_minus_topfile):
        os.makedirs(minus_plus_plus_minus_topfile)
    if file_name in os.listdir(minus_plus_plus_minus_topfile):
        minus_p1_plus_p2_plus_p3_minus_p4 = pickle.load(open(minus_plus_plus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_plus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=l2, label='-++-')
        pickle.dump(minus_p1_plus_p2_plus_p3_minus_p4, open(minus_plus_plus_minus_topfile + file_name, 'wb'))

    minus_plus_minus_minus_topfile = oracle_canidates_file + '-+--/'
    if not os.path.exists(minus_plus_minus_minus_topfile):
        os.makedirs(minus_plus_minus_minus_topfile)
    if file_name in os.listdir(minus_plus_minus_minus_topfile):
        minus_p1_plus_p2_minus_p3_minus_p4 = pickle.load(open(minus_plus_minus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_minus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=l2, label='-+--')
        pickle.dump(minus_p1_plus_p2_minus_p3_minus_p4, open(minus_plus_minus_minus_topfile + file_name, 'wb'))

    minus_minus_plus_minus_topfile = oracle_canidates_file + '--+-/'
    if not os.path.exists(minus_minus_plus_minus_topfile):
        os.makedirs(minus_minus_plus_minus_topfile)
    if file_name in os.listdir(minus_minus_plus_minus_topfile):
        minus_p1_minus_p2_plus_p3_minus_p4 = pickle.load(open(minus_minus_plus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_plus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=l2, label='--+-')
        pickle.dump(minus_p1_minus_p2_plus_p3_minus_p4, open(minus_minus_plus_minus_topfile + file_name, 'wb'))

    minus_minus_minus_minus_topfile = oracle_canidates_file + '----/'
    if not os.path.exists(minus_minus_minus_minus_topfile):
        os.makedirs(minus_minus_minus_minus_topfile)
    if file_name in os.listdir(minus_minus_minus_minus_topfile):
        minus_p1_minus_p2_minus_p3_minus_p4 = pickle.load(open(minus_minus_minus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_minus_p3_minus_p4 = sqlodbc.get_p1_p2_and_p3_p4_by_e1_e2(e1=e1, e2=l2, label='----')
        pickle.dump(minus_p1_minus_p2_minus_p3_minus_p4, open(minus_minus_minus_minus_topfile + file_name, 'wb'))

    return plus_p1_plus_p2_plus_p3_minus_p4, plus_p1_plus_p2_minus_p3_minus_p4, \
           plus_p1_minus_p2_plus_p3_minus_p4, plus_p1_minus_p2_minus_p3_minus_p4, \
           minus_p1_plus_p2_plus_p3_minus_p4, minus_p1_plus_p2_minus_p3_minus_p4, \
           minus_p1_minus_p2_plus_p3_minus_p4, minus_p1_minus_p2_minus_p3_minus_p4


"""4-2"""


def get_hop4_2_paths_by_entity(e1, bi_direction=False):
    '''+++'''
    file_name = e1
    plus_plus_plus_plus_topfile = oracle_canidates_file + '++++/'
    if not os.path.exists(plus_plus_plus_plus_topfile):
        os.makedirs(plus_plus_plus_plus_topfile)
    if file_name in os.listdir(plus_plus_plus_plus_topfile):
        plus_p1_plus_p2_plus_p3_plus_p4 = pickle.load(open(plus_plus_plus_plus_topfile + file_name, 'rb'))
    else:
        plus_p1_plus_p2_plus_p3_plus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=e1, label='++++')
        pickle.dump(plus_p1_plus_p2_plus_p3_plus_p4, open(plus_plus_plus_plus_topfile + file_name, 'wb'))
    if not bi_direction:
        return plus_p1_plus_p2_plus_p3_plus_p4

    plus_plus_plus_minus_topfile = oracle_canidates_file + '+++-/'
    if not os.path.exists(plus_plus_plus_minus_topfile):
        os.makedirs(plus_plus_plus_minus_topfile)
    if file_name in os.listdir(plus_plus_plus_minus_topfile):
        plus_p1_plus_p2_plus_p3_minus_p4 = pickle.load(open(plus_plus_plus_minus_topfile + file_name, 'rb'))
    else:
        plus_p1_plus_p2_plus_p3_minus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=e1, label='+++-')
        pickle.dump(plus_p1_plus_p2_plus_p3_minus_p4, open(plus_plus_plus_minus_topfile + file_name, 'wb'))

    plus_plus_minus_plus_topfile = oracle_canidates_file + '++-+/'
    if not os.path.exists(plus_plus_minus_plus_topfile):
        os.makedirs(plus_plus_minus_plus_topfile)
    if file_name in os.listdir(plus_plus_minus_plus_topfile):
        plus_p1_plus_p2_minus_p3_plus_p4 = pickle.load(open(plus_plus_minus_plus_topfile + file_name, 'rb'))
    else:
        plus_p1_plus_p2_minus_p3_plus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=e1, label='++-+')
        pickle.dump(plus_p1_plus_p2_minus_p3_plus_p4, open(plus_plus_minus_plus_topfile + file_name, 'wb'))

    plus_plus_minus_minus_topfile = oracle_canidates_file + '++--/'
    if not os.path.exists(plus_plus_minus_minus_topfile):
        os.makedirs(plus_plus_minus_minus_topfile)
    if file_name in os.listdir(plus_plus_minus_minus_topfile):
        plus_p1_plus_p2_minus_p3_minus_p4 = pickle.load(open(plus_plus_minus_minus_topfile + file_name, 'rb'))
    else:
        plus_p1_plus_p2_minus_p3_minus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=e1, label='++--')
        pickle.dump(plus_p1_plus_p2_minus_p3_minus_p4, open(plus_plus_minus_minus_topfile + file_name, 'wb'))

    plus_minus_plus_plus_topfile = oracle_canidates_file + '+-++/'
    if not os.path.exists(plus_minus_plus_plus_topfile):
        os.makedirs(plus_minus_plus_plus_topfile)
    if file_name in os.listdir(plus_minus_plus_plus_topfile):
        plus_p1_minus_p2_plus_p3_plus_p4 = pickle.load(open(plus_minus_plus_plus_topfile + file_name, 'rb'))
    else:
        plus_p1_minus_p2_plus_p3_plus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=e1, label='+-++')
        pickle.dump(plus_p1_minus_p2_plus_p3_plus_p4, open(plus_minus_plus_plus_topfile + file_name, 'wb'))

    plus_minus_plus_minus_topfile = oracle_canidates_file + '+-+-/'
    if not os.path.exists(plus_minus_plus_minus_topfile):
        os.makedirs(plus_minus_plus_minus_topfile)
    if file_name in os.listdir(plus_minus_plus_minus_topfile):
        plus_p1_minus_p2_plus_p3_minus_p4 = pickle.load(open(plus_minus_plus_minus_topfile + file_name, 'rb'))
    else:
        plus_p1_minus_p2_plus_p3_minus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=e1, label='+-+-')
        pickle.dump(plus_p1_minus_p2_plus_p3_minus_p4, open(plus_minus_plus_minus_topfile + file_name, 'wb'))

    plus_minus_minus_plus_topfile = oracle_canidates_file + '+--+/'
    if not os.path.exists(plus_minus_minus_plus_topfile):
        os.makedirs(plus_minus_minus_plus_topfile)
    if file_name in os.listdir(plus_minus_minus_plus_topfile):
        plus_p1_minus_p2_minus_p3_plus_p4 = pickle.load(open(plus_minus_minus_plus_topfile + file_name, 'rb'))
    else:
        plus_p1_minus_p2_minus_p3_plus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=e1, label='+--+')
        pickle.dump(plus_p1_minus_p2_minus_p3_plus_p4, open(plus_minus_minus_plus_topfile + file_name, 'wb'))

    plus_minus_minus_minus_topfile = oracle_canidates_file + '+---/'
    if not os.path.exists(plus_minus_minus_minus_topfile):
        os.makedirs(plus_minus_minus_minus_topfile)
    if file_name in os.listdir(plus_minus_minus_minus_topfile):
        plus_p1_minus_p2_minus_p3_minus_p4 = pickle.load(open(plus_minus_minus_minus_topfile + file_name, 'rb'))
    else:
        plus_p1_minus_p2_minus_p3_minus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=e1, label='+---')
        pickle.dump(plus_p1_minus_p2_minus_p3_minus_p4, open(plus_minus_minus_minus_topfile + file_name, 'wb'))

    minus_plus_plus_plus_topfile = oracle_canidates_file + '-+++/'
    if not os.path.exists(minus_plus_plus_plus_topfile):
        os.makedirs(minus_plus_plus_plus_topfile)
    if file_name in os.listdir(minus_plus_plus_plus_topfile):
        minus_p1_plus_p2_plus_p3_plus_p4 = pickle.load(open(minus_plus_plus_plus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_plus_p3_plus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=e1, label='-+++')
        pickle.dump(minus_p1_plus_p2_plus_p3_plus_p4, open(minus_plus_plus_plus_topfile + file_name, 'wb'))

    minus_plus_plus_minus_topfile = oracle_canidates_file + '-++-/'
    if not os.path.exists(minus_plus_plus_minus_topfile):
        os.makedirs(minus_plus_plus_minus_topfile)
    if file_name in os.listdir(minus_plus_plus_minus_topfile):
        minus_p1_plus_p2_plus_p3_minus_p4 = pickle.load(open(minus_plus_plus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_plus_p3_minus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=e1, label='-++-')
        pickle.dump(minus_p1_plus_p2_plus_p3_minus_p4, open(minus_plus_plus_minus_topfile + file_name, 'wb'))

    minus_plus_minus_plus_topfile = oracle_canidates_file + '-+-+/'
    if not os.path.exists(minus_plus_minus_plus_topfile):
        os.makedirs(minus_plus_minus_plus_topfile)
    if file_name in os.listdir(minus_plus_minus_plus_topfile):
        minus_p1_plus_p2_minus_p3_plus_p4 = pickle.load(open(minus_plus_minus_plus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_minus_p3_plus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=e1, label='-+-+')
        pickle.dump(minus_p1_plus_p2_minus_p3_plus_p4, open(minus_plus_minus_plus_topfile + file_name, 'wb'))

    minus_plus_minus_minus_topfile = oracle_canidates_file + '-+--/'
    if not os.path.exists(minus_plus_minus_minus_topfile):
        os.makedirs(minus_plus_minus_minus_topfile)
    if file_name in os.listdir(minus_plus_minus_minus_topfile):
        minus_p1_plus_p2_minus_p3_minus_p4 = pickle.load(open(minus_plus_minus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_minus_p3_minus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=e1, label='-+--')
        pickle.dump(minus_p1_plus_p2_minus_p3_minus_p4, open(minus_plus_minus_minus_topfile + file_name, 'wb'))

    minus_minus_plus_plus_topfile = oracle_canidates_file + '--++/'
    if not os.path.exists(minus_minus_plus_plus_topfile):
        os.makedirs(minus_minus_plus_plus_topfile)
    if file_name in os.listdir(minus_minus_plus_plus_topfile):
        minus_p1_minus_p2_plus_p3_plus_p4 = pickle.load(open(minus_minus_plus_plus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_plus_p3_plus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=e1, label='--++')
        pickle.dump(minus_p1_minus_p2_plus_p3_plus_p4, open(minus_minus_plus_plus_topfile + file_name, 'wb'))

    minus_minus_plus_minus_topfile = oracle_canidates_file + '--+-/'
    if not os.path.exists(minus_minus_plus_minus_topfile):
        os.makedirs(minus_minus_plus_minus_topfile)
    if file_name in os.listdir(minus_minus_plus_minus_topfile):
        minus_p1_minus_p2_plus_p3_minus_p4 = pickle.load(open(minus_minus_plus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_plus_p3_minus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=e1, label='--+-')
        pickle.dump(minus_p1_minus_p2_plus_p3_minus_p4, open(minus_minus_plus_minus_topfile + file_name, 'wb'))

    minus_minus_minus_plus_topfile = oracle_canidates_file + '---+/'
    if not os.path.exists(minus_minus_minus_plus_topfile):
        os.makedirs(minus_minus_minus_plus_topfile)
    if file_name in os.listdir(minus_minus_minus_plus_topfile):
        minus_p1_minus_p2_minus_p3_plus_p4 = pickle.load(open(minus_minus_minus_plus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_minus_p3_plus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=e1, label='---+')
        pickle.dump(minus_p1_minus_p2_minus_p3_plus_p4, open(minus_minus_minus_plus_topfile + file_name, 'wb'))

    minus_minus_minus_minus_topfile = oracle_canidates_file + '----/'
    if not os.path.exists(minus_minus_minus_minus_topfile):
        os.makedirs(minus_minus_minus_minus_topfile)
    if file_name in os.listdir(minus_minus_minus_minus_topfile):
        minus_p1_minus_p2_minus_p3_minus_p4 = pickle.load(open(minus_minus_minus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_minus_p3_minus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=e1, label='----')
        pickle.dump(minus_p1_minus_p2_minus_p3_minus_p4, open(minus_minus_minus_minus_topfile + file_name, 'wb'))

    return plus_p1_plus_p2_plus_p3_plus_p4, plus_p1_plus_p2_plus_p3_minus_p4, plus_p1_plus_p2_minus_p3_plus_p4, plus_p1_plus_p2_minus_p3_minus_p4, \
           plus_p1_minus_p2_plus_p3_plus_p4, plus_p1_minus_p2_plus_p3_minus_p4, plus_p1_minus_p2_minus_p3_plus_p4, plus_p1_minus_p2_minus_p3_minus_p4, \
           minus_p1_plus_p2_plus_p3_plus_p4, minus_p1_plus_p2_plus_p3_minus_p4, minus_p1_plus_p2_minus_p3_plus_p4, minus_p1_plus_p2_minus_p3_minus_p4, \
           minus_p1_minus_p2_plus_p3_plus_p4, minus_p1_minus_p2_plus_p3_minus_p4, minus_p1_minus_p2_minus_p3_plus_p4, minus_p1_minus_p2_minus_p3_minus_p4


def get_hop4_2_paths_by_literal(l1, bi_direction=False):
    '''-+++'''
    file_name = l1
    minus_plus_plus_plus_topfile = oracle_canidates_file + '-+++/'
    if not os.path.exists(minus_plus_plus_plus_topfile):
        os.makedirs(minus_plus_plus_plus_topfile)
    if file_name in os.listdir(minus_plus_plus_plus_topfile):
        minus_p1_plus_p2_plus_p3_plus_p4 = pickle.load(open(minus_plus_plus_plus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_plus_p3_plus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=l1, label='-+++')
        pickle.dump(minus_p1_plus_p2_plus_p3_plus_p4, open(minus_plus_plus_plus_topfile + file_name, 'wb'))
    if not bi_direction:
        return minus_p1_plus_p2_plus_p3_plus_p4

    minus_plus_plus_minus_topfile = oracle_canidates_file + '-++-/'
    if not os.path.exists(minus_plus_plus_minus_topfile):
        os.makedirs(minus_plus_plus_minus_topfile)
    if file_name in os.listdir(minus_plus_plus_minus_topfile):
        minus_p1_plus_p2_plus_p3_minus_p4 = pickle.load(open(minus_plus_plus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_plus_p3_minus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=l1, label='-++-')
        pickle.dump(minus_p1_plus_p2_plus_p3_minus_p4, open(minus_plus_plus_minus_topfile + file_name, 'wb'))

    minus_plus_minus_plus_topfile = oracle_canidates_file + '-+-+/'
    if not os.path.exists(minus_plus_minus_plus_topfile):
        os.makedirs(minus_plus_minus_plus_topfile)
    if file_name in os.listdir(minus_plus_minus_plus_topfile):
        minus_p1_plus_p2_minus_p3_plus_p4 = pickle.load(open(minus_plus_minus_plus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_minus_p3_plus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=l1, label='-+-+')
        pickle.dump(minus_p1_plus_p2_minus_p3_plus_p4, open(minus_plus_minus_plus_topfile + file_name, 'wb'))

    minus_plus_minus_minus_topfile = oracle_canidates_file + '-+--/'
    if not os.path.exists(minus_plus_minus_minus_topfile):
        os.makedirs(minus_plus_minus_minus_topfile)
    if file_name in os.listdir(minus_plus_minus_minus_topfile):
        minus_p1_plus_p2_minus_p3_minus_p4 = pickle.load(open(minus_plus_minus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_plus_p2_minus_p3_minus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=l1, label='-+--')
        pickle.dump(minus_p1_plus_p2_minus_p3_minus_p4, open(minus_plus_minus_minus_topfile + file_name, 'wb'))

    minus_minus_plus_plus_topfile = oracle_canidates_file + '--++/'
    if not os.path.exists(minus_minus_plus_plus_topfile):
        os.makedirs(minus_minus_plus_plus_topfile)
    if file_name in os.listdir(minus_minus_plus_plus_topfile):
        minus_p1_minus_p2_plus_p3_plus_p4 = pickle.load(open(minus_minus_plus_plus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_plus_p3_plus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=l1, label='--++')
        pickle.dump(minus_p1_minus_p2_plus_p3_plus_p4, open(minus_minus_plus_plus_topfile + file_name, 'wb'))

    minus_minus_plus_minus_topfile = oracle_canidates_file + '--+-/'
    if not os.path.exists(minus_minus_plus_minus_topfile):
        os.makedirs(minus_minus_plus_minus_topfile)
    if file_name in os.listdir(minus_minus_plus_minus_topfile):
        minus_p1_minus_p2_plus_p3_minus_p4 = pickle.load(open(minus_minus_plus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_plus_p3_minus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=l1, label='--+-')
        pickle.dump(minus_p1_minus_p2_plus_p3_minus_p4, open(minus_minus_plus_minus_topfile + file_name, 'wb'))

    minus_minus_minus_plus_topfile = oracle_canidates_file + '---+/'
    if not os.path.exists(minus_minus_minus_plus_topfile):
        os.makedirs(minus_minus_minus_plus_topfile)
    if file_name in os.listdir(minus_minus_minus_plus_topfile):
        minus_p1_minus_p2_minus_p3_plus_p4 = pickle.load(open(minus_minus_minus_plus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_minus_p3_plus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=l1, label='---+')
        pickle.dump(minus_p1_minus_p2_minus_p3_plus_p4, open(minus_minus_minus_plus_topfile + file_name, 'wb'))

    minus_minus_minus_minus_topfile = oracle_canidates_file + '----/'
    if not os.path.exists(minus_minus_minus_minus_topfile):
        os.makedirs(minus_minus_minus_minus_topfile)
    if file_name in os.listdir(minus_minus_minus_minus_topfile):
        minus_p1_minus_p2_minus_p3_minus_p4 = pickle.load(open(minus_minus_minus_minus_topfile + file_name, 'rb'))
    else:
        minus_p1_minus_p2_minus_p3_minus_p4 = sqlodbc.get_p1_p2_p3_p4_by_e1(s=l1, label='----')
        pickle.dump(minus_p1_minus_p2_minus_p3_minus_p4, open(minus_minus_minus_minus_topfile + file_name, 'wb'))

    return minus_p1_plus_p2_plus_p3_plus_p4, minus_p1_plus_p2_plus_p3_minus_p4, minus_p1_plus_p2_minus_p3_plus_p4, minus_p1_plus_p2_minus_p3_minus_p4, \
           minus_p1_minus_p2_plus_p3_plus_p4, minus_p1_minus_p2_plus_p3_minus_p4, minus_p1_minus_p2_minus_p3_plus_p4, minus_p1_minus_p2_minus_p3_minus_p4

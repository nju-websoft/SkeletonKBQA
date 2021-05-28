from datasets_interface.virtuoso_interface import freebase_kb_interface
import collections
from method_sp.grounding.grounding_args import mediators_instances_set


# 读取1步路径
def get_s_p_p_o_bylinkedentity(entity):
    '''获得entity两个方向一步path,得到entity的s_p, p_o'''
    p_o, o_set, p1_set = freebase_kb_interface.get_p_o_by_entity(entity)
    s_p, s_set, p_set = freebase_kb_interface.get_s_p_by_entity(entity)
    return s_p, p_o


# 读取2步mediator路径
def get_p1_mediator_p2_answer(entity, filter_class_mid=None):
    '''获得一个entity structure0_0的候选，即entity-p-a, entity-p1-mediator-p2-a
    entity 出发走两步, 并且第二个节点是mediator实体'''
    s1_p1, p1_o1 = get_s_p_p_o_bylinkedentity(entity)
    o1_set, o1_p1_dict = get_o1s_o1_p1(p1_o1)
    o1set_s_p_dict, o1set_p_o_dict = get_mediator_relation_entity(o1_set, filter_class_mid)
    return generate_p1_cvt_p2_answer(o1_p1_dict, o1set_p_o_dict, 0)


# 读取2步cvt(所有节点)的路径
def get_p1_cvt_p2_answer(entity):
    '''得到entity的p1_cvt_p2_answer
    :return pathes
    '''
    s1_p1, p1_o1 = get_s_p_p_o_bylinkedentity(entity)
    o1_set, o1_p1_dict = get_o1s_o1_p1(p1_o1)
    o1set_s_p, o1set_p_o = get_cvt_relation_entity(o1_set)
    return generate_p1_cvt_p2_answer(o1_p1_dict, o1set_p_o, 0)


# utils
def get_mediator_relation_entity(candidatecvts, filter_class_mid=[]):
    '''如果candidatecvts中的cvt, 如果是mediators的实例的话，就在走一步，
    获得mediator实体连的relation和entity， 并且度数<=1824的'''
    cvt_p_o = dict()
    cvt_s_p = dict()
    if filter_class_mid is None:
        filter_class_mid = ['m.05zppz', 'm.02zsn']
    for cvt in candidatecvts:
        if cvt in mediators_instances_set and cvt not in filter_class_mid:
            p_o_set, o_set, p1_set = freebase_kb_interface.get_p_o_by_entity(cvt)
            s_p_set, s_set, p2_set = freebase_kb_interface.get_s_p_by_entity(cvt)
            cvt_p_o[cvt] = p_o_set
            cvt_s_p[cvt] = s_p_set
    return cvt_s_p, cvt_p_o


# utils
def get_cvt_relation_entity(candidatecvts):
    '''获得除了结构1时cvt连得relation和entity;
    这里的cvt就是正常实体, 得到正常实体所连的relation和answer，对正常实体做了一定过滤要求所连之和小于1824'''
    cvt_p_o = dict()
    cvt_s_p = dict()
    for cvt in candidatecvts:
        p_o_set, o_set, p1_set = freebase_kb_interface.get_p_o_by_entity(cvt)
        s_p_set, s_set, p2_set = freebase_kb_interface.get_s_p_by_entity(cvt)
        cvt_p_o[cvt] = p_o_set
        cvt_s_p[cvt] = s_p_set
    return cvt_s_p, cvt_p_o


# utils
def get_o1s_o1_p1(p1_o1):
    '''将p1_o1写成以o1为key的dictionary'''
    o1s = set()
    o1_p1 = collections.defaultdict(set)
    for one in p1_o1:
        p1,o1 =one.split("\t")
        o1_p1[o1].add(p1)
        o1s.add(o1)
    return o1s, o1_p1


# utils
def generate_p1_cvt_p2_answer(cvt_p1_dict, cvt_p2_entity, reverse):
    '''拼接: cvt两部分所连拼接 将cvt_p1和cvt_p2_entity连城p1_cvt_p2_answer '''
    p1_cvt_p2_answer = set()
    for cvt in cvt_p2_entity:
        p1s = cvt_p1_dict[cvt]
        p2_entity_set = cvt_p2_entity[cvt]
        for p1 in p1s:
            for p2_entity in p2_entity_set:
                if reverse == 0:
                    p1_cvt_p2_answer.add("\t".join([p1, cvt, p2_entity]))
                else:
                    answer, p2 = p2_entity.split("\t")
                    p1_cvt_p2_answer.add("\t".join([p1, cvt, p2, answer]))
    return list(p1_cvt_p2_answer)


def get_s_p_by_literal_none(literal_value):
    s_p_set, s_set, p_set = freebase_kb_interface.get_s_p_literal_none(literal_value)
    return s_p_set

import collections
from common_structs.grounded_graph import GroundedNode, GroundedEdge


def parser_comparative_q_freebase_ir(data_dict, s1=None,t1=None):
    candidate_graphquery_list = []
    for querytype in data_dict:
        # if querytype == "3_0":
        #     candidate_graphquery_list.extend(_3_0_to_graphs(data_dict['1_0'], s1=s1, t1=t1))
        if querytype == "3_0_1":
            candidate_graphquery_list.extend(_3_0_1_to_graphs(data_dict['3_0_1'], s1=s1, t1=t1))
        elif querytype == "3_0_2":
            candidate_graphquery_list.extend(_3_0_2_to_graphs(data_dict['3_0_2'], s1=s1, t1=t1))
        # elif querytype == "3_1":
        #     candidate_graphquery_list.extend(_3_1_to_graphs(data_dict['1_2_2'], s1=s1, t1=t1))
        elif querytype == "3_1_2":
            candidate_graphquery_list.extend(_3_1_2_to_graphs(data_dict['3_1_2'], s1=s1, t1=t1))
        elif querytype == "3_1_3":
            candidate_graphquery_list.extend(_3_1_3_to_graphs(data_dict['3_1_3'], s1=s1, t1=t1))
    return candidate_graphquery_list


def _3_0_to_graphs(candidate_pathes, s1, t1):
    '''1_0 	entity-{p}->o	对应, 第1位对应到路径是p, 第二位对应到路径是o
    ns:m.0dhqrm "organization.organization.headquarters\tm.08cshk7'''
    candidate_graphquery_list = []
    current_nid = 1
    node_topic_entity = GroundedNode(nid=current_nid, node_type=t1, id=s1, type_class='', friendly_name="", question_node=0)
    current_nid+=1
    node_answer_entity= GroundedNode(nid=current_nid, node_type="class", id='?a', type_class='', friendly_name="", question_node=1)
    p_answers=collections.defaultdict(set)
    for candidate in candidate_pathes:
        cols = candidate.split("\t")
        if len(cols) != 2:
            continue
        relation, answer_entity = cols
        p_answers[relation].add(answer_entity)

    for p in p_answers:
        candidate_graphquery = dict()
        candidate_graphquery["querytype"] = '3_0'
        candidate_graphquery["nodes"] = [node_topic_entity, node_answer_entity]
        edge = GroundedEdge(start=node_topic_entity.nid, end=node_answer_entity.nid, relation=p)
        candidate_graphquery["edges"] = [edge]
        candidate_graphquery["path"] = p
        candidate_graphquery["denotation"] = list(p_answers[p])
        candidate_graphquery_list.append(candidate_graphquery)
    return candidate_graphquery_list


def _3_0_1_to_graphs(candidate_pathes, s1, t1):
    '''
        e-{p1}->a*-{p2}->literal
        "location.location.contains\tm.05vd5\tlocation.location.area\t47.0",
    '''
    candidate_graphquery_list = []
    current_nid = 1
    node_topic_entity = GroundedNode(nid=current_nid, node_type=t1, id=s1, type_class='', friendly_name="", question_node=0)
    current_nid += 1
    node_answer_entity = GroundedNode(nid=current_nid, node_type="class", id='?a', type_class='', friendly_name="", question_node=1)
    current_nid += 1
    node_literal = GroundedNode(nid=current_nid, node_type="literal", id='?literal', type_class='', friendly_name="", question_node=0)
    p1_p2_answers = collections.defaultdict(set)
    for candidate in candidate_pathes:
        cols = candidate.split("\t")
        if len(cols) != 4:
            continue
        p1, answer_entity, p2, literal_entity = cols
        p1_p2_answers['\t'.join([p1, p2])].add((answer_entity, literal_entity))

    for p1_p2 in p1_p2_answers:
        candidate_graphquery = dict()
        candidate_graphquery["querytype"] = '3_0_1'
        candidate_graphquery["nodes"] = [node_topic_entity, node_answer_entity, node_literal]
        p1, p2 = p1_p2.split('\t')
        edge1 = GroundedEdge(start=node_topic_entity.nid, end=node_answer_entity.nid, relation=p1)
        edge2 = GroundedEdge(start=node_answer_entity.nid, end=node_literal.nid, relation=p2)
        candidate_graphquery["edges"] = [edge1, edge2]
        candidate_graphquery["path"] = p1_p2
        candidate_graphquery["denotation"] = list(p1_p2_answers[p1_p2])
        candidate_graphquery_list.append(candidate_graphquery)
    return candidate_graphquery_list


def _3_0_2_to_graphs(candidate_pathes, s1, t1):
    '''
        #e-{p1}->*a-{p2}->*m-{p3}->literal 对应
        "location.location.contains\tm.06s9y\tlocation.statistical_region.gdp_real\tm.0hnzhpd\tmeasurement_unit.adjusted_money_value.adjusted_value\t192244189.0
        :param paths:
        :param s1:
        :param t1:
        :return:
        '''
    candidate_graphquery_list = []
    current_nid = 1
    node_topic_entity = GroundedNode(nid=current_nid, node_type=t1, id=s1, type_class='', friendly_name="",question_node=0)
    current_nid += 1
    node_answer_entity = GroundedNode(nid=current_nid, node_type="class", id='?a', type_class='', friendly_name="", question_node=1)
    current_nid += 1
    node_m_entity = GroundedNode(nid=current_nid, node_type="class", id='?m', type_class='', friendly_name="",question_node=0)
    current_nid += 1
    node_literal = GroundedNode(nid=current_nid, node_type="literal", id='?literal', type_class='', friendly_name="", question_node=0)
    p1_p2_p3_answers = collections.defaultdict(set)
    for candidate in candidate_pathes:
        cols = candidate.split('\t')
        if len(cols) != 6:
            continue
        p1, answer_entity, p2, m_entity, p3, literal_entity = candidate.split("\t")
        p1_p2_p3_answers['\t'.join([p1, p2, p3])].add((answer_entity, literal_entity))

    for p1_p2_p3 in p1_p2_p3_answers:
        candidate_graphquery = dict()
        candidate_graphquery["querytype"] = '3_0_2'
        candidate_graphquery["nodes"] = [node_topic_entity, node_answer_entity, node_m_entity, node_literal]
        p1, p2, p3 = p1_p2_p3.split('\t')
        edge1 = GroundedEdge(start=node_topic_entity.nid, end=node_answer_entity.nid, relation=p1)
        edge2 = GroundedEdge(start=node_answer_entity.nid, end=node_m_entity.nid, relation=p2)
        edge3 = GroundedEdge(start=node_m_entity.nid, end=node_literal.nid, relation=p3)
        candidate_graphquery["edges"] = [edge1, edge2, edge3]
        candidate_graphquery["path"] = p1_p2_p3
        candidate_graphquery["denotation"] = list(p1_p2_p3_answers[p1_p2_p3])
        candidate_graphquery_list.append(candidate_graphquery)
    return candidate_graphquery_list


def _3_1_to_graphs(candidate_pathes, s1, t1):
    '''
    e-{p1}->m-{p2}->a
    "user.tsegaran.random.taxonomy_subject.entry\tm.04_8c54\tuser.tsegaran.random.taxonomy_entry.taxonomy\tm.04n6k",
    '''
    candidate_graphquery_list = []
    current_nid = 1
    node_topic_entity = GroundedNode(nid=current_nid, node_type=t1, id=s1, type_class='', friendly_name="",question_node=0)
    current_nid += 1
    node_m_entity = GroundedNode(nid=current_nid, node_type="class", id='?m', type_class='', friendly_name="",question_node=0)
    current_nid += 1
    node_answer_entity = GroundedNode(nid=current_nid, node_type="class", id='?a', type_class='', friendly_name="", question_node=1)
    p1_p2_answers = collections.defaultdict(set)
    for candidate in candidate_pathes:
        cols = candidate.split("\t")
        if len(cols) != 4:
            continue
        p1, m_entity, p2, answer_entity = cols
        p1_p2_answers['\t'.join([p1, p2])].add(answer_entity)
    for p1_p2 in p1_p2_answers:
        candidate_graphquery = dict()
        candidate_graphquery["querytype"] = '3_1'
        candidate_graphquery["nodes"] = [node_topic_entity, node_m_entity, node_answer_entity]
        p1, p2 = p1_p2.split('\t')
        edge1 = GroundedEdge(start=node_topic_entity.nid, end=node_m_entity.nid, relation=p1)
        edge2 = GroundedEdge(start=node_m_entity.nid, end=node_answer_entity.nid, relation=p2)
        candidate_graphquery["edges"] = [edge1, edge2]
        candidate_graphquery["path"] = p1_p2
        candidate_graphquery["denotation"] = list(p1_p2_answers[p1_p2])
        candidate_graphquery_list.append(candidate_graphquery)
    return candidate_graphquery_list


def _3_1_2_to_graphs(candidate_pathes, s1, t1):
    '''
        #e-{p1}->m-{p2}->a-{p3}->literal 对应
        "location.location.contains\tm.047tj\tgeography.island.body_of_water\tm.05rgl\tgeography.body_of_water.surface_area\t165200000.0"
        :param paths:
        :param s1:
        :param t1:
        :return:
        '''
    candidate_graphquery_list = []
    current_nid = 1
    node_topic_entity = GroundedNode(nid=current_nid, node_type=t1, id=s1, type_class='', friendly_name="",question_node=0)
    current_nid += 1
    node_m_entity = GroundedNode(nid=current_nid, node_type="class", id='?m', type_class='', friendly_name="",question_node=0)
    current_nid += 1
    node_answer_entity = GroundedNode(nid=current_nid, node_type="class", id='?a', type_class='', friendly_name="", question_node=1)
    current_nid += 1
    node_literal_entity = GroundedNode(nid=current_nid, node_type="literal", id='?literal', type_class='', friendly_name="",question_node=0)
    p1_p2_p3_answers = collections.defaultdict(set)
    for candidate in candidate_pathes:
        cols = candidate.split('\t')
        if len(cols) != 6:
            continue
        p1, m_entity, p2, answer_entity, p3, literal_entity = candidate.split("\t")
        p1_p2_p3_answers['\t'.join([p1, p2, p3])].add((answer_entity, literal_entity))

    for p1_p2_p3 in p1_p2_p3_answers:
        candidate_graphquery = dict()
        candidate_graphquery["querytype"] = '3_1_2'
        candidate_graphquery["nodes"] = [node_topic_entity, node_m_entity, node_answer_entity, node_literal_entity]
        p1, p2, p3 = p1_p2_p3.split('\t')
        edge1 = GroundedEdge(start=node_topic_entity.nid, end=node_m_entity.nid, relation=p1)
        edge2 = GroundedEdge(start=node_m_entity.nid, end=node_answer_entity.nid, relation=p2)
        edge3 = GroundedEdge(start=node_answer_entity.nid, end=node_literal_entity.nid, relation=p3)
        candidate_graphquery["edges"] = [edge1, edge2, edge3]
        candidate_graphquery["path"] = p1_p2_p3
        candidate_graphquery["denotation"] = list(p1_p2_p3_answers[p1_p2_p3])
        candidate_graphquery_list.append(candidate_graphquery)
    return candidate_graphquery_list


def _3_1_3_to_graphs(candidate_pathes, s1, t1):
    '''
        #e-{p1}->m1-{p2}->a-{p3}->m2->{p4}->literal 对应
        location.location.contains\tm.047tj\tgeography.island.body_of_water\tm.05rgl\tlocation.location.geolocation\tm.05l1d9y\tlocation.geocode.latitude\t0.0",
        :param paths:
        :param s1:
        :param t1:
        :return:
    '''
    candidate_graphquery_list = []
    current_nid = 1
    node_topic_entity = GroundedNode(nid=current_nid, node_type=t1, id=s1, type_class='', friendly_name="", question_node=0)
    current_nid += 1
    node_m_entity = GroundedNode(nid=current_nid, node_type="class", id='?m', type_class='', friendly_name="", question_node=0)
    current_nid += 1
    node_answer_entity = GroundedNode(nid=current_nid, node_type="class", id='?a', type_class='', friendly_name="",question_node=1)
    current_nid += 1
    node_c_entity = GroundedNode(nid=current_nid, node_type="class", id='?c', type_class='', friendly_name="",question_node=0)
    current_nid += 1
    node_literal_entity = GroundedNode(nid=current_nid, node_type="literal", id='?literal', type_class='', friendly_name="",question_node=0)

    p1_p2_p3_p4_answers = collections.defaultdict(set)
    for candidate in candidate_pathes:
        cols = candidate.split('\t')
        if len(cols) != 8:
            continue
        p1, m_1_entity, p2, answer_entity, p3, m_2_entity, p4, literal_entity = candidate.split("\t")
        p1_p2_p3_p4_answers['\t'.join([p1, p2, p3, p4])].add((answer_entity, literal_entity))

    for p1_p2_p3_p4 in p1_p2_p3_p4_answers:
        candidate_graphquery = dict()
        candidate_graphquery["querytype"] = '3_1_3'
        candidate_graphquery["nodes"] = [node_topic_entity, node_m_entity, node_answer_entity, node_c_entity, node_literal_entity]
        p1, p2, p3, p4 = p1_p2_p3_p4.split('\t')
        edge1 = GroundedEdge(start=node_topic_entity.nid, end=node_m_entity.nid, relation=p1)
        edge2 = GroundedEdge(start=node_m_entity.nid, end=node_answer_entity.nid, relation=p2)
        edge3 = GroundedEdge(start=node_answer_entity.nid, end=node_c_entity.nid, relation=p3)
        edge4 = GroundedEdge(start=node_c_entity.nid, end=node_literal_entity.nid, relation=p4)
        candidate_graphquery["edges"] = [edge1, edge2, edge3, edge4]
        candidate_graphquery["path"] = p1_p2_p3_p4
        candidate_graphquery["denotation"] = list(p1_p2_p3_p4_answers[p1_p2_p3_p4])
        candidate_graphquery_list.append(candidate_graphquery)
    return candidate_graphquery_list

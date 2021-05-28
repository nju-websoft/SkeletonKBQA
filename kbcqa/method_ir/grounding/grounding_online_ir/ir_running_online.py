from common.hand_files import write_json

from method_ir.grounding.grounding_online_ir import ir_online_utils, ir_candidate_generation_kb
from method_ir.grounding.grounding_online_ir.candidate_generation_args import consider_literal, has_bi_direction


def get_hop1(entity_or_literal_with_type_list):
    '''
    :param entity_or_literal: (entity, 'type')  type:entity, literal  example: [['m.01_2n', 'entity']], [['m.015lwh', 'literal']]
    :return: hop1 dict
    '''
    def get_hop1_0_path(entity_or_literal_with_type_list):
        hop1_0 = dict()
        if len(entity_or_literal_with_type_list) != 2:
            return hop1_0
        e1, e2 = ir_online_utils.get_entity1_and_entity2_from_list(entity_or_literal_with_type_list)
        if has_bi_direction:
            plus_p1, minus_p2 = ir_candidate_generation_kb.get_1_0_path_by_entities(e1=e1, e2=e2, bi_direction=has_bi_direction)
            hop1_0['+'] = list(plus_p1)
            hop1_0['-'] = list(minus_p2)
        else:
            plus_p1 = ir_candidate_generation_kb.get_1_0_path_by_entities(e1=e1, e2=e2, bi_direction=has_bi_direction)
            hop1_0['+'] = list(plus_p1)
        return hop1_0

    def get_hop1_1_path(entity_or_literal_with_type_list):
        hop1 = dict()
        if len(entity_or_literal_with_type_list) != 1:
            return hop1
        entity_or_literal_with_type = entity_or_literal_with_type_list[0]
        if entity_or_literal_with_type[1] == 'literal' and consider_literal:
            s_p_set, s_set, p_set = ir_candidate_generation_kb.get_s_p_by_literal(entity_or_literal_with_type[0])
            hop1['-'] = list(p_set)
        else: #entity
            p_o_set, o_set, p1_set = ir_candidate_generation_kb.get_p_o_by_entity(entity_or_literal_with_type[0])
            hop1['+'] = list(p1_set)
            if has_bi_direction:
                s_p_set, s_set, p2_set = ir_candidate_generation_kb.get_s_p_by_entity(entity_or_literal_with_type[0])
                hop1['-'] = list(p2_set)
        return hop1

    hops1 = dict()
    hops1['1_0'] = get_hop1_0_path(entity_or_literal_with_type_list=entity_or_literal_with_type_list)
    hops1['1_1'] = get_hop1_1_path(entity_or_literal_with_type_list=entity_or_literal_with_type_list)
    return hops1


def get_hop2(entity_or_literal_with_type_list):
    '''
    :param entity_or_literal: (entity, 'type')  type:entity, literal
    example: [['m.01_2n', 'entity'], ['m.015lwh', 'literal']]
    :return: hop2 dict
    '''
    def _get_hop2_1_path(entity_or_literal_with_type_list):
        hop2_path_dict = dict()
        if len(entity_or_literal_with_type_list) != 1:
            return hop2_path_dict
        entity_or_literal_with_type = entity_or_literal_with_type_list[0]
        if entity_or_literal_with_type[1] == 'literal' and consider_literal:
            if has_bi_direction:
                minus_p1_plus_p2, minus_p1_minus_p2 = ir_candidate_generation_kb.get_hop2_1_path_by_literal(literal_value=entity_or_literal_with_type[0], bi_direction=has_bi_direction)
                hop2_path_dict['-+'] = minus_p1_plus_p2
                hop2_path_dict['--'] = minus_p1_minus_p2
            else:
                minus_p1_plus_p2 = ir_candidate_generation_kb.get_hop2_1_path_by_literal(literal_value=entity_or_literal_with_type[0], bi_direction=has_bi_direction)
                hop2_path_dict['-+'] = minus_p1_plus_p2
        else: # entity
            if has_bi_direction:
                plus_p1_plus_p2, plus_p1_minus_p2, minus_p1_plus_p2, minus_p1_minus_p2 = ir_candidate_generation_kb.get_hop2_1_path_by_entity(entity=entity_or_literal_with_type[0], bi_direction=has_bi_direction)  # ++, +-
                hop2_path_dict['++'] = plus_p1_plus_p2
                hop2_path_dict['+-'] = plus_p1_minus_p2
                hop2_path_dict['-+'] = minus_p1_plus_p2
                hop2_path_dict['--'] = minus_p1_minus_p2
            else:
                plus_p1_plus_p2 = ir_candidate_generation_kb.get_hop2_1_path_by_entity(entity=entity_or_literal_with_type[0], bi_direction=has_bi_direction) # ++
                hop2_path_dict['++'] = plus_p1_plus_p2
        return hop2_path_dict

    def _get_hop2_2_star(entity_or_literal_with_type_list):
        hop2_star_dict = dict()
        if len(entity_or_literal_with_type_list) != 2:
            return hop2_star_dict
        literal_num = ir_online_utils.get_literal_num(entity_or_literal_with_type_list)
        if literal_num == 1 and consider_literal: # one literal, one entity
            l1, e2 = ir_online_utils.get_literal_and_entity_from_list(entity_or_literal_with_type_list)
            if has_bi_direction:
                minus_p1_plus_p2, minus_p1_minus_p2 = ir_candidate_generation_kb.get_hop2_2_star_by_entity_literal(l1=l1, e2=e2, bi_direction=has_bi_direction)
                hop2_star_dict['-+'] = minus_p1_plus_p2
                hop2_star_dict['--'] = minus_p1_minus_p2
            else:
                hop2_star_dict['-+'] = ir_candidate_generation_kb.get_hop2_2_star_by_entity_literal(l1=l1, e2=e2, bi_direction=has_bi_direction)

        elif literal_num == 2 and consider_literal: # two literal, zero entity
            l1, l2 = ir_online_utils.get_entity1_and_entity2_from_list(entity_or_literal_with_type_list)
            hop2_star_dict['--'] = ir_candidate_generation_kb.get_hop2_2_star_by_literals(l1=l1, l2=l2)

        else: # zero literal, two entity
            e1, e2 = ir_online_utils.get_entity1_and_entity2_from_list(entity_or_literal_with_type_list)
            if has_bi_direction:
                plus_p1_plus_p2, plus_p1_minus_p2, minus_p1_plus_p2, minus_p1_minus_p2 = ir_candidate_generation_kb.get_hop2_2_star_by_entites(e1=e1, e2=e2, bi_direction=True)
                hop2_star_dict['++'] = plus_p1_plus_p2
                hop2_star_dict['+-'] = plus_p1_minus_p2
                hop2_star_dict['-+'] = minus_p1_plus_p2
                hop2_star_dict['--'] = minus_p1_minus_p2
            else:
                hop2_star_dict['++'] = ir_candidate_generation_kb.get_hop2_2_star_by_entites(e1=e1, e2=e2, bi_direction=False)
        return hop2_star_dict

    hop2 = dict()
    if len(entity_or_literal_with_type_list) == 1:
        hop2['2_1'] = _get_hop2_1_path(entity_or_literal_with_type_list)
    elif len(entity_or_literal_with_type_list) == 2:
        hop2['2_2'] = _get_hop2_2_star(entity_or_literal_with_type_list)
    return hop2


def get_hop3(entity_or_literal_with_type_list):

    def _get_hop3_path(entity_or_literal_with_type_list):
        hop3_path_dict = dict()
        if len(entity_or_literal_with_type_list) != 1:
            return hop3_path_dict
        entity_or_literal_with_type = entity_or_literal_with_type_list[0]
        if entity_or_literal_with_type[1] == 'literal' and consider_literal:
            if has_bi_direction:
                minus_p1_plus_p2_plus_p3, minus_p1_plus_p2_minus_p3, minus_p1_minus_p2_plus_p3, minus_p1_minus_p2_minus_p3 = ir_candidate_generation_kb.get_hop3_3_path_by_literal(
                    l1=entity_or_literal_with_type[0], bi_direction=has_bi_direction)
                hop3_path_dict['-++'] = minus_p1_plus_p2_plus_p3
                hop3_path_dict['-+-'] = minus_p1_plus_p2_minus_p3
                hop3_path_dict['--+'] = minus_p1_minus_p2_plus_p3
                hop3_path_dict['---'] = minus_p1_minus_p2_minus_p3
            else:
                hop3_path_dict['-++'] = ir_candidate_generation_kb.get_hop3_3_path_by_literal(l1=entity_or_literal_with_type[0], bi_direction=has_bi_direction)
        else:
            if has_bi_direction:
                plus_p1_plus_p2_plus_p3, plus_p1_plus_p2_minus_p3, plus_p1_minus_p2_plus_p3, plus_p1_minus_p2_minus_p3, \
                minus_p1_plus_p2_plus_p3, minus_p1_plus_p2_minus_p3, minus_p1_minus_p2_plus_p3, minus_p1_minus_p2_minus_p3 = ir_candidate_generation_kb.get_hop3_3_path_by_entity(
                    e1=entity_or_literal_with_type[0], bi_direction=has_bi_direction)
                hop3_path_dict['+++'] = plus_p1_plus_p2_plus_p3
                hop3_path_dict['++-'] = plus_p1_plus_p2_minus_p3
                hop3_path_dict['+-+'] = plus_p1_minus_p2_plus_p3
                hop3_path_dict['+--'] = plus_p1_minus_p2_minus_p3
                hop3_path_dict['-++'] = minus_p1_plus_p2_plus_p3
                hop3_path_dict['-+-'] = minus_p1_plus_p2_minus_p3
                hop3_path_dict['--+'] = minus_p1_minus_p2_plus_p3
                hop3_path_dict['---'] = minus_p1_minus_p2_minus_p3
            else:
                hop3_path_dict['+++'] = ir_candidate_generation_kb.get_hop3_3_path_by_entity(e1=entity_or_literal_with_type[0], bi_direction=has_bi_direction)
        return hop3_path_dict

    def _get_hop3_star(entity_or_literal_with_type_list):

        def __get_hop3_1_star_zero_literal(e1, e2):
            '''插入点在el的边上'''
            hop3_star_dict = dict()
            if has_bi_direction:
                plus_p1_plus_p2_plus_p3, plus_p1_plus_p2_minus_p3, plus_p1_minus_p2_plus_p3, plus_p1_minus_p2_minus_p3,\
                minus_p1_plus_p2_plus_p3, minus_p1_plus_p2_minus_p3, minus_p1_minus_p2_plus_p3, minus_p1_minus_p2_minus_p3 = ir_candidate_generation_kb.get_hop3_1_star_by_entities(
                    e1=e1, e2=e2, bi_direction=has_bi_direction)
                hop3_star_dict['+++'] = plus_p1_plus_p2_plus_p3
                hop3_star_dict['++-'] = plus_p1_plus_p2_minus_p3
                hop3_star_dict['+-+'] = plus_p1_minus_p2_plus_p3
                hop3_star_dict['+--'] = plus_p1_minus_p2_minus_p3
                hop3_star_dict['-++'] = minus_p1_plus_p2_plus_p3
                hop3_star_dict['-+-'] = minus_p1_plus_p2_minus_p3
                hop3_star_dict['--+'] = minus_p1_minus_p2_plus_p3
                hop3_star_dict['---'] = minus_p1_minus_p2_minus_p3
            else:
                # plus_p1_plus_p2_plus_p3
                hop3_star_dict['+++'] = ir_candidate_generation_kb.get_hop3_1_star_by_entities(e1=e1, e2=e2, bi_direction=has_bi_direction)
            return hop3_star_dict

        def __get_hop3_2_star_zero_literal(e1, e2):
            '''插入点在e2的边上'''
            hop3_star_dict = dict()
            if has_bi_direction:
                plus_p1_plus_p2_plus_p3, plus_p1_plus_p2_minus_p3, plus_p1_minus_p2_plus_p3, plus_p1_minus_p2_minus_p3,\
                minus_p1_plus_p2_plus_p3, minus_p1_plus_p2_minus_p3, minus_p1_minus_p2_plus_p3, minus_p1_minus_p2_minus_p3 = ir_candidate_generation_kb.get_hop3_2_star_by_entities(
                    e1=e1, e2=e2, bi_direction=has_bi_direction)
                hop3_star_dict['+++'] = plus_p1_plus_p2_plus_p3
                hop3_star_dict['++-'] = plus_p1_plus_p2_minus_p3
                hop3_star_dict['+-+'] = plus_p1_minus_p2_plus_p3
                hop3_star_dict['+--'] = plus_p1_minus_p2_minus_p3
                hop3_star_dict['-++'] = minus_p1_plus_p2_plus_p3
                hop3_star_dict['-+-'] = minus_p1_plus_p2_minus_p3
                hop3_star_dict['--+'] = minus_p1_minus_p2_plus_p3
                hop3_star_dict['---'] = minus_p1_minus_p2_minus_p3
            else:
                # plus_p1_plus_p2_plus_p3
                hop3_star_dict['+++'] = ir_candidate_generation_kb.get_hop3_2_star_by_entities(e1=e1, e2=e2, bi_direction=has_bi_direction)
            return hop3_star_dict

        def __get_hop3_1_star_one_literal(e1, l2):
            '''插入点在l1的边上'''
            hop3_star_dict = dict()
            if has_bi_direction:
                plus_p1_plus_p2_minus_p3, plus_p1_minus_p2_minus_p3, minus_p1_plus_p2_minus_p3, minus_p1_minus_p2_minus_p3 \
                    = ir_candidate_generation_kb.get_hop3_1_star_by_entity_literal(e1=e1, l2=l2, bi_direction=has_bi_direction)
                hop3_star_dict['++-'] = plus_p1_plus_p2_minus_p3
                hop3_star_dict['+--'] = plus_p1_minus_p2_minus_p3
                hop3_star_dict['-+-'] = minus_p1_plus_p2_minus_p3
                hop3_star_dict['---'] = minus_p1_minus_p2_minus_p3
            else:
                plus_p1_plus_p2_minus_p3 = ir_candidate_generation_kb.get_hop3_1_star_by_entity_literal(e1=e1, l2=l2, bi_direction=has_bi_direction)
                hop3_star_dict['++-'] = plus_p1_plus_p2_minus_p3
            return hop3_star_dict

        def __get_hop3_2_star_one_literal(e1, l2):
            '''插入点在e2的边上'''
            hop3_star_dict = dict()
            if has_bi_direction:
                plus_p1_plus_p2_minus_p3, plus_p1_minus_p2_minus_p3, minus_p1_plus_p2_minus_p3, minus_p1_minus_p2_minus_p3 \
                    = ir_candidate_generation_kb.get_hop3_2_star_by_entity_literal(e1=e1, l2=l2, bi_direction=has_bi_direction)
                hop3_star_dict['++-'] = plus_p1_plus_p2_minus_p3
                hop3_star_dict['+--'] = plus_p1_minus_p2_minus_p3
                hop3_star_dict['-+-'] = minus_p1_plus_p2_minus_p3
                hop3_star_dict['---'] = minus_p1_minus_p2_minus_p3
            else:
                plus_p1_plus_p2_minus_p3 = ir_candidate_generation_kb.get_hop3_2_star_by_entity_literal(e1=e1, l2=l2, bi_direction=has_bi_direction)
                hop3_star_dict['++-'] = plus_p1_plus_p2_minus_p3
            return hop3_star_dict

        def __get_hop3_1_star_two_literal(l1, l2):
            '''插入点在l1的边上'''
            hop3_star_dict = dict()
            if has_bi_direction:
                minus_p1_plus_p2_minus_p3, minus_p1_minus_p2_minus_p3 = ir_candidate_generation_kb.get_hop3_1_star_by_literals(l1=l1, l2=l2, bi_direction=has_bi_direction)
                hop3_star_dict['-+-'] = minus_p1_plus_p2_minus_p3
                hop3_star_dict['---'] = minus_p1_minus_p2_minus_p3
            else:
                hop3_star_dict['-+-'] = ir_candidate_generation_kb.get_hop3_1_star_by_literals(l1=l1, l2=l2, bi_direction=has_bi_direction)
            return hop3_star_dict

        def __get_hop3_2_star_two_literal(l1, l2):
            '''插入点在l2的边上'''
            hop3_star_dict = dict()
            if has_bi_direction:
                minus_p1_plus_p2_minus_p3, minus_p1_minus_p2_minus_p3 = ir_candidate_generation_kb.get_hop3_2_star_by_literals(l1=l1, l2=l2, bi_direction=has_bi_direction)
                hop3_star_dict['-+-'] = minus_p1_plus_p2_minus_p3
                hop3_star_dict['---'] = minus_p1_minus_p2_minus_p3
            else:
                hop3_star_dict['-+-'] = ir_candidate_generation_kb.get_hop3_2_star_by_literals(l1=l1, l2=l2, bi_direction=has_bi_direction)
            return hop3_star_dict

        hop3_star_dict = dict()
        if len(entity_or_literal_with_type_list) != 2:
            return hop3_star_dict
        literal_num = ir_online_utils.get_literal_num(entity_or_literal_with_type_list)
        if len(literal_num) == 1 and consider_literal: # one literal, one entity
            e1, l2 = ir_online_utils.get_literal_and_entity_from_list(entity_or_literal_with_type_list)
            hop3_star_dict['3_1'] = __get_hop3_1_star_one_literal(e1=e1, l2=l2)
            hop3_star_dict['3_2'] = __get_hop3_2_star_one_literal(e1=e1, l2=l2)
        elif len(literal_num) == 2 and consider_literal:  # two literal, zero entity
            l1, l2 = ir_online_utils.get_entity1_and_entity2_from_list(entity_or_literal_with_type_list)
            hop3_star_dict['3_1'] = __get_hop3_1_star_two_literal(l1=l1, l2=l2)
            hop3_star_dict['3_2'] = __get_hop3_2_star_two_literal(l1=l1, l2=l2)
        else:   # zero literal, two entity
            e1, e2 = ir_online_utils.get_entity1_and_entity2_from_list(entity_or_literal_with_type_list)
            hop3_star_dict['3_1'] = __get_hop3_1_star_zero_literal(e1=e1, e2=e2)
            hop3_star_dict['3_2'] = __get_hop3_2_star_zero_literal(e1=e1, e2=e2)
        return hop3_star_dict

    hop3 = dict()
    if len(entity_or_literal_with_type_list) == 1:
        hop3['3_3'] = _get_hop3_path(entity_or_literal_with_type_list)
    elif len(entity_or_literal_with_type_list) == 2:
        hop3 = _get_hop3_star(entity_or_literal_with_type_list)
        # hop3 = generate_utils.merge(hop3, temp_dict)
    return hop3


def get_hop4(entity_or_literal_with_type_list):

    def _get_hop4_1_star(entity_or_literal_with_type_list):

        def __get_hop4_1_star_zero_literal(e1, e2):
            '''分别在el and el的边上均插入点'''
            hop4_1_star = dict()
            if has_bi_direction:
                plus_p1_plus_p2_plus_p3_plus_p4, plus_p1_plus_p2_plus_p3_minus_p4, plus_p1_plus_p2_minus_p3_plus_p4, plus_p1_plus_p2_minus_p3_minus_p4, \
                        plus_p1_minus_p2_plus_p3_plus_p4, plus_p1_minus_p2_plus_p3_minus_p4, plus_p1_minus_p2_minus_p3_plus_p4, plus_p1_minus_p2_minus_p3_minus_p4, \
                        minus_p1_plus_p2_plus_p3_plus_p4, minus_p1_plus_p2_plus_p3_minus_p4, minus_p1_plus_p2_minus_p3_plus_p4, minus_p1_plus_p2_minus_p3_minus_p4, \
                        minus_p1_minus_p2_plus_p3_plus_p4, minus_p1_minus_p2_plus_p3_minus_p4, minus_p1_minus_p2_minus_p3_plus_p4, minus_p1_minus_p2_minus_p3_minus_p4 = \
                    ir_candidate_generation_kb.get_hop4_1_star_by_entities(e1=e1, e2=e2, bi_direction=has_bi_direction)
                hop4_1_star['++++'] = plus_p1_plus_p2_plus_p3_plus_p4
                hop4_1_star['+++-'] = plus_p1_plus_p2_plus_p3_minus_p4
                hop4_1_star['++-+'] = plus_p1_plus_p2_minus_p3_plus_p4
                hop4_1_star['++--'] = plus_p1_plus_p2_minus_p3_minus_p4

                hop4_1_star['+-++'] = plus_p1_minus_p2_plus_p3_plus_p4
                hop4_1_star['+-+-'] = plus_p1_minus_p2_plus_p3_minus_p4
                hop4_1_star['+--+'] = plus_p1_minus_p2_minus_p3_plus_p4
                hop4_1_star['+---'] = plus_p1_minus_p2_minus_p3_minus_p4

                hop4_1_star['-+++'] = minus_p1_plus_p2_plus_p3_plus_p4
                hop4_1_star['-++-'] = minus_p1_plus_p2_plus_p3_minus_p4
                hop4_1_star['-+-+'] = minus_p1_plus_p2_minus_p3_plus_p4
                hop4_1_star['-+--'] = minus_p1_plus_p2_minus_p3_minus_p4

                hop4_1_star['--++'] = minus_p1_minus_p2_plus_p3_plus_p4
                hop4_1_star['--+-'] = minus_p1_minus_p2_plus_p3_minus_p4
                hop4_1_star['---+'] = minus_p1_minus_p2_minus_p3_plus_p4
                hop4_1_star['----'] = minus_p1_minus_p2_minus_p3_minus_p4
            else:
                plus_p1_plus_p2_plus_p3_plus_p4 = ir_candidate_generation_kb.get_hop4_1_star_by_entities(e1=e1, e2=e2, bi_direction=has_bi_direction)
                hop4_1_star['++++'] = plus_p1_plus_p2_plus_p3_plus_p4
            return hop4_1_star

        def __get_hop4_1_star_one_literal(e1, l2):
            '''分别在l1 and el的边上均插入点'''
            hop4_1_star = dict()
            if has_bi_direction:
                pass
                plus_p1_plus_p2_plus_p3_minus_p4, plus_p1_plus_p2_minus_p3_minus_p4, plus_p1_minus_p2_plus_p3_minus_p4,plus_p1_minus_p2_minus_p3_minus_p4, \
                minus_p1_plus_p2_plus_p3_minus_p4, minus_p1_plus_p2_minus_p3_minus_p4, minus_p1_minus_p2_plus_p3_minus_p4, minus_p1_minus_p2_minus_p3_minus_p4 \
                    = ir_candidate_generation_kb.get_hop4_1_star_by_entity_literal(e1=e1, l2=l2, bi_direction=has_bi_direction)
                hop4_1_star['+++-'] = plus_p1_plus_p2_plus_p3_minus_p4
                hop4_1_star['++--'] = plus_p1_plus_p2_minus_p3_minus_p4
                hop4_1_star['+-+-'] = plus_p1_minus_p2_plus_p3_minus_p4
                hop4_1_star['+---'] = plus_p1_minus_p2_minus_p3_minus_p4
                hop4_1_star['-++-'] = minus_p1_plus_p2_plus_p3_minus_p4
                hop4_1_star['-+--'] = minus_p1_plus_p2_minus_p3_minus_p4
                hop4_1_star['--+-'] = minus_p1_minus_p2_plus_p3_minus_p4
                hop4_1_star['----'] = minus_p1_minus_p2_minus_p3_minus_p4
            else:
                plus_p1_plus_p2_plus_p3_minus_p4 = ir_candidate_generation_kb.get_hop4_1_star_by_entity_literal(e1=e1, l2=l2, bi_direction=has_bi_direction)
                hop4_1_star['+++-'] = plus_p1_plus_p2_plus_p3_minus_p4
            return hop4_1_star

        def __get_hop4_1_star_two_literal(l1, l2):
            '''分别在l1 and el的边上均插入点'''
            hop4_1_star = dict()
            if has_bi_direction:
                minus_p1_plus_p2_plus_p3_minus_p4, minus_p1_plus_p2_minus_p3_minus_p4, \
                minus_p1_minus_p2_plus_p3_minus_p4, minus_p1_minus_p2_minus_p3_minus_p4 = ir_candidate_generation_kb.get_hop4_1_star_by_literals(l1=l1, l2=l2, bi_direction=has_bi_direction)
                hop4_1_star['-++-'] = minus_p1_plus_p2_plus_p3_minus_p4
                hop4_1_star['-+--'] = minus_p1_plus_p2_minus_p3_minus_p4
                hop4_1_star['--+-'] = minus_p1_minus_p2_plus_p3_minus_p4
                hop4_1_star['----'] = minus_p1_minus_p2_minus_p3_minus_p4
            else:
                minus_p1_plus_p2_plus_p3_minus_p4 = ir_candidate_generation_kb.get_hop4_1_star_by_literals(l1=l1, l2=l2, bi_direction=has_bi_direction)
                hop4_1_star['-++-'] = minus_p1_plus_p2_plus_p3_minus_p4
            return hop4_1_star

        hop4_star_dict = dict()
        if len(entity_or_literal_with_type_list) != 2:
            return hop4_star_dict
        literal_num = ir_online_utils.get_literal_num(entity_or_literal_with_type_list)
        if len(literal_num) == 1 and consider_literal:  # one literal, one entity
            e1, l2 = ir_online_utils.get_literal_and_entity_from_list(entity_or_literal_with_type_list)
            hop4_star_dict = __get_hop4_1_star_one_literal(e1=e1, l2=l2)
        elif len(literal_num) == 2 and consider_literal:  # two literal, zero entity
            l1, l2 = ir_online_utils.get_entity1_and_entity2_from_list(entity_or_literal_with_type_list)
            hop4_star_dict = __get_hop4_1_star_two_literal(l1=l1, l2=l2)
        else:  # zero literal, two entity
            e1, e2 = ir_online_utils.get_entity1_and_entity2_from_list(entity_or_literal_with_type_list)
            hop4_star_dict = __get_hop4_1_star_zero_literal(e1=e1, e2=e2)
        return hop4_star_dict

    def _get_hop4_2_path(entity_or_literal_with_type_list):
        hop4_path_dict = dict()
        if len(entity_or_literal_with_type_list) != 1:
            return hop4_path_dict
        entity_or_literal_with_type = entity_or_literal_with_type_list[0]
        if entity_or_literal_with_type[1] == 'literal' and consider_literal:
            if has_bi_direction:
                minus_p1_plus_p2_plus_p3_plus_p4, minus_p1_plus_p2_plus_p3_minus_p4, minus_p1_plus_p2_minus_p3_plus_p4, minus_p1_plus_p2_minus_p3_minus_p4, \
                minus_p1_minus_p2_plus_p3_plus_p4, minus_p1_minus_p2_plus_p3_minus_p4, minus_p1_minus_p2_minus_p3_plus_p4, minus_p1_minus_p2_minus_p3_minus_p4 = \
                    ir_candidate_generation_kb.get_hop4_2_paths_by_literal(l1=entity_or_literal_with_type[0], bi_direction=has_bi_direction)
                hop4_path_dict['-+++'] = minus_p1_plus_p2_plus_p3_plus_p4
                hop4_path_dict['-++-'] = minus_p1_plus_p2_plus_p3_minus_p4
                hop4_path_dict['-+-+'] = minus_p1_plus_p2_minus_p3_plus_p4
                hop4_path_dict['--++'] = minus_p1_plus_p2_minus_p3_minus_p4
                hop4_path_dict['-+--'] = minus_p1_minus_p2_plus_p3_plus_p4
                hop4_path_dict['--+-'] = minus_p1_minus_p2_plus_p3_minus_p4
                hop4_path_dict['---+'] = minus_p1_minus_p2_minus_p3_plus_p4
                hop4_path_dict['----'] = minus_p1_minus_p2_minus_p3_minus_p4
            else:
                minus_p1_plus_p2_plus_p3_plus_p4, minus_p1_plus_p2_plus_p3_minus_p4, minus_p1_plus_p2_minus_p3_plus_p4, minus_p1_plus_p2_minus_p3_minus_p4, \
                minus_p1_minus_p2_plus_p3_plus_p4, minus_p1_minus_p2_plus_p3_minus_p4, minus_p1_minus_p2_minus_p3_plus_p4, minus_p1_minus_p2_minus_p3_minus_p4 = \
                    ir_candidate_generation_kb.get_hop4_2_paths_by_literal(l1=entity_or_literal_with_type[0], bi_direction=has_bi_direction)
                hop4_path_dict['-+++'] = minus_p1_plus_p2_plus_p3_plus_p4

        else: # entity
            if has_bi_direction:
                plus_p1_plus_p2_plus_p3_plus_p4, plus_p1_plus_p2_plus_p3_minus_p4, plus_p1_plus_p2_minus_p3_plus_p4, plus_p1_plus_p2_minus_p3_minus_p4, \
                plus_p1_minus_p2_plus_p3_plus_p4, plus_p1_minus_p2_plus_p3_minus_p4, plus_p1_minus_p2_minus_p3_plus_p4, plus_p1_minus_p2_minus_p3_minus_p4, \
                minus_p1_plus_p2_plus_p3_plus_p4, minus_p1_plus_p2_plus_p3_minus_p4, minus_p1_plus_p2_minus_p3_plus_p4, minus_p1_plus_p2_minus_p3_minus_p4, \
                minus_p1_minus_p2_plus_p3_plus_p4, minus_p1_minus_p2_plus_p3_minus_p4, minus_p1_minus_p2_minus_p3_plus_p4, minus_p1_minus_p2_minus_p3_minus_p4 = \
                    ir_candidate_generation_kb.get_hop4_2_paths_by_entity(e1=entity_or_literal_with_type[0])

                hop4_path_dict['++++'] = plus_p1_plus_p2_plus_p3_plus_p4
                hop4_path_dict['+++-'] = plus_p1_plus_p2_plus_p3_minus_p4
                hop4_path_dict['++-+'] = plus_p1_plus_p2_minus_p3_plus_p4
                hop4_path_dict['++--'] = plus_p1_plus_p2_minus_p3_minus_p4

                hop4_path_dict['+-++'] = plus_p1_minus_p2_plus_p3_plus_p4
                hop4_path_dict['+-+-'] = plus_p1_minus_p2_plus_p3_minus_p4
                hop4_path_dict['+--+'] = plus_p1_minus_p2_minus_p3_plus_p4
                hop4_path_dict['+---'] = plus_p1_minus_p2_minus_p3_minus_p4

                hop4_path_dict['-+++'] = minus_p1_plus_p2_plus_p3_plus_p4
                hop4_path_dict['-++-'] = minus_p1_plus_p2_plus_p3_minus_p4
                hop4_path_dict['-+-+'] = minus_p1_plus_p2_minus_p3_plus_p4
                hop4_path_dict['-+--'] = minus_p1_plus_p2_minus_p3_minus_p4

                hop4_path_dict['--++'] = minus_p1_minus_p2_plus_p3_plus_p4
                hop4_path_dict['--+-'] = minus_p1_minus_p2_plus_p3_minus_p4
                hop4_path_dict['---+'] = minus_p1_minus_p2_minus_p3_plus_p4
                hop4_path_dict['----'] = minus_p1_minus_p2_minus_p3_minus_p4

            else:
                plus_p1_plus_p2_plus_p3_plus_p4 = ir_candidate_generation_kb.get_hop4_2_paths_by_entity(e1=entity_or_literal_with_type[0])
                hop4_path_dict['++++'] = plus_p1_plus_p2_plus_p3_plus_p4

        return hop4_path_dict

    hop4 = dict()
    if len(entity_or_literal_with_type_list) == 1:
        hop4['4_2'] = _get_hop4_2_path(entity_or_literal_with_type_list)
    elif len(entity_or_literal_with_type_list) == 2:
        hop4['4_1'] = _get_hop4_1_star(entity_or_literal_with_type_list)
    return hop4


def _get_hop1_hop2_by_online(topic_entities_with_types):
    hop1 = get_hop1(entity_or_literal_with_type_list=topic_entities_with_types)
    hop2 = get_hop2(entity_or_literal_with_type_list=topic_entities_with_types)
    hop1_list = []
    for hop1_type, hop1_path in hop1.items():
        for label, path_list in hop1_path.items():
            for path in path_list:
                temp_dict = dict()
                temp_dict['path_type'] = hop1_type
                predicate_list = path.split('\t')
                new_predicate_list = []
                for i in range(len(predicate_list)):
                    new_predicate_list.append(label[i])
                    new_predicate_list.append(predicate_list[i])
                temp_dict['path'] = new_predicate_list
                hop1_list.append(temp_dict)
    hop2_list = []
    for hop2_type, hop2_path in hop2.items():
        for label, path_list in hop2_path.items():
            for path in path_list:
                temp_dict = dict()
                temp_dict['path_type'] = hop2_type
                predicate_list = path.split('\t')
                new_predicate_list = []
                for i in range(len(predicate_list)):
                    new_predicate_list.append(label[i])
                    new_predicate_list.append(predicate_list[i])
                temp_dict['path'] = new_predicate_list
                hop2_list.append(temp_dict)
    return hop1_list, hop2_list


""""""


def run_cwq(data_type, output_file):
    from datasets_interface.question_interface import complexwebquestion_interface
    ann_data_list = []
    complexwebq_struct_list = []
    if data_type == 'train':
        complexwebq_struct_list = complexwebquestion_interface.complexwebq_train_list
    elif data_type == 'test':
        complexwebq_struct_list = complexwebquestion_interface.complexwebq_test_list
    elif data_type == 'dev':
        complexwebq_struct_list = complexwebquestion_interface.complexwebq_dev_list
    for i, complexwebq_struct in enumerate(complexwebq_struct_list):
        question_normal = complexwebq_struct.question
        print(complexwebq_struct.ID)
        entities_list = complexwebquestion_interface.get_topic_entities_by_question(question_normal)
        abstract_question = complexwebquestion_interface.get_abstract_question_by_question(question=question_normal)
        parsed_sparql = complexwebq_struct.parsed_sparql
        sparql = complexwebq_struct.sparql
        gold_triples = ir_online_utils.get_triples_by_sparql_json(parsed_sparql)
        gold_path = ir_online_utils.convert_triples_to_path(triples=gold_triples)
        gold = {}
        gold['question_type'] = ir_online_utils.get_question_type_by_sparql_json(sparql_json=parsed_sparql)
        gold['topic_entities'] = entities_list
        gold['aggregation_function'] = ir_online_utils.get_aggregation_function_by_sparql_json(sparq_json=parsed_sparql)
        gold['type_constraints'] = ir_online_utils.get_type_constraints_by_sparql_json(sparql_json=parsed_sparql)
        gold['gold_path'] = gold_path
        gold['gold_triples'] = gold_triples
        gold['sparql'] = sparql
        data = {
            'qid': complexwebq_struct.ID,
            'question_normal': question_normal,
            'abstract_question': abstract_question,
            'gold': gold,
            'no_positive_path': True,
            'hop1': [],
            'hop2': []
        }
        topic_entities_with_types = ir_online_utils.topic_entities_with_t(entities_list=entities_list)
        # hop1, hop2 = _get_hop1_hop2_by_enum_grounded_graphs(topic_entities_with_types)
        hop1, hop2 = _get_hop1_hop2_by_online(topic_entities_with_types=topic_entities_with_types)
        data['hop1'] = hop1
        data['hop2'] = hop2
        data['no_positive_path'] = ir_online_utils.is_exist_gold_path(hop_list=hop1 + hop2, gold_path=gold_path)
        ann_data_list.append(data)
        break
    write_json(ann_data_list, output_file)


def run_graphquestions(data_type, output_file):
    from datasets_interface.question_interface import graphquestion_interface
    ann_data_list = []
    graphq_struct_list = []
    if data_type == 'train':
        graphq_struct_list = graphquestion_interface.train_graph_questions_struct
    elif data_type == 'test':
        graphq_struct_list = graphquestion_interface.test_graph_questions_struct
    for i, graphq_struct in enumerate(graphq_struct_list):
        question_normal = graphq_struct.question
        print(graphq_struct.qid)
        entities_list = graphquestion_interface.get_topic_entities_by_question(question_normal)
        abstract_question = graphquestion_interface.get_abstract_question_by_question(question=question_normal)
        parsed_sparql = graphq_struct.parsed_sparql
        sparql = graphq_struct.sparql_query
        # gold_triples = generate_utils.get_triples_by_sparql_json(parsed_sparql)
        gold_triples = ir_online_utils.get_triples_by_grounded_graph_edges(nodes=graphq_struct.nodes, edges=graphq_struct.edges)
        gold_path = ir_online_utils.convert_triples_to_path(triples=gold_triples)
        gold = {}
        gold['question_type'] = ir_online_utils.get_question_type_by_sparql_json(sparql_json=parsed_sparql)
        gold['topic_entities'] = entities_list
        gold['aggregation_function'] = ir_online_utils.get_aggregation_function_by_sparql_json(sparq_json=parsed_sparql)
        gold['type_constraints'] = ir_online_utils.get_type_constraints_by_sparql_json(sparql_json=parsed_sparql)
        gold['gold_path'] = gold_path
        gold['gold_triples'] = gold_triples
        gold['sparql'] = sparql
        data = {
            'qid': graphq_struct.qid,
            'question_normal': graphq_struct.question,
            'abstract_question': abstract_question,
            'gold': gold,
            'no_positive_path': True,
            'hop1': [],
            'hop2': []
        }
        topic_entities_with_types = ir_online_utils.topic_entities_with_t(entities_list=entities_list)
        # hop1, hop2 = _get_hop1_hop2_by_enum_grounded_graphs(topic_entities_with_types)
        hop1, hop2 = _get_hop1_hop2_by_online(topic_entities_with_types)
        data['hop1'] = hop1
        data['hop2'] = hop2
        data['no_positive_path'] = ir_online_utils.is_exist_gold_path(hop_list=hop1 + hop2, gold_path=gold_path)
        ann_data_list.append(data)
    write_json(ann_data_list, output_file)


def run_lcquad(data_type, output_file):
    from datasets_interface.question_interface import lcquad_1_0_interface
    ann_data_list = []
    lcquad_list = []
    if data_type == 'train':
        lcquad_list = lcquad_1_0_interface.lcquad_train_list
    elif data_type == 'test':
        lcquad_list = lcquad_1_0_interface.lcquad_test_list
    for i, lcquad_struct in enumerate(lcquad_list):
        question_normal = lcquad_struct.question_normal
        print(lcquad_struct.qid)
        entities_list = lcquad_1_0_interface.get_topic_entities_by_question(question_normal)
        abstract_question = lcquad_1_0_interface.get_abstract_question_by_question(question=question_normal)
        parsed_sparql = lcquad_struct.parsed_sparql
        sparql = lcquad_struct.sparql
        gold_triples = ir_online_utils.get_triples_by_sparql_json(parsed_sparql)
        gold_path = ir_online_utils.convert_triples_to_path(triples=gold_triples)
        gold = {}
        gold['question_type'] = ir_online_utils.get_question_type_by_sparql_json(sparql_json=parsed_sparql)
        gold['topic_entities'] = entities_list
        gold['aggregation_function'] = ir_online_utils.get_aggregation_function_by_sparql_json(sparq_json=parsed_sparql)
        gold['type_constraints'] = ir_online_utils.get_type_constraints_by_sparql_json(sparql_json=parsed_sparql)
        gold['gold_path'] = gold_path
        gold['gold_triples'] = gold_triples
        gold['sparql'] = sparql
        data = {
            'qid': lcquad_struct.qid,
            'question_normal': lcquad_struct.question_normal,
            'abstract_question': abstract_question,
            'gold': gold,
            'no_positive_path': True,
            'hop1': [],
            'hop2': []
        }
        topic_entities_with_types = ir_online_utils.topic_entities_with_t(entities_list=entities_list)
        # hop1, hop2 = _get_hop1_hop2_by_enum_grounded_graphs(topic_entities_with_types)
        try:
            hop1, hop2 = _get_hop1_hop2_by_online(topic_entities_with_types=topic_entities_with_types)
        except Exception as e:
            hop1, hop2 = [], []
            print('Error:\t', question_normal)
        data['hop1'] = hop1
        data['hop2'] = hop2
        data['no_positive_path'] = ir_online_utils.is_exist_gold_path(hop_list=hop1 + hop2, gold_path=gold_path)
        ann_data_list.append(data)
    write_json(ann_data_list, output_file)


""""""


def test_one_entity():
    """
    [['m.0d3k14', 'entity']]
    [['en.harlem_renaissance', 'entity']]
    [['en.palomar_observatory', 'entity'], ['en.reflecting_telescope', 'entity']]
    :return:
    """

    n_hops = 1
    entity_or_literal_with_type_list = [['m.0_tlmng', 'entity']] #m.02bb47

    hops = dict()
    if n_hops == 1:
        hops = get_hop1(entity_or_literal_with_type_list=entity_or_literal_with_type_list)
    elif n_hops == 2:
        hop1 = get_hop1(entity_or_literal_with_type_list=entity_or_literal_with_type_list)
        hop2 = get_hop2(entity_or_literal_with_type_list=entity_or_literal_with_type_list)
        hops = ir_online_utils.merge(hop1, hop2)
    elif n_hops == 3:
        get_hop1(entity_or_literal_with_type_list=entity_or_literal_with_type_list)
        get_hop2(entity_or_literal_with_type_list=entity_or_literal_with_type_list)
        get_hop3(entity_or_literal_with_type_list=entity_or_literal_with_type_list)
    elif n_hops == 4:
        get_hop1(entity_or_literal_with_type_list=entity_or_literal_with_type_list)
        get_hop2(entity_or_literal_with_type_list=entity_or_literal_with_type_list)
        get_hop3(entity_or_literal_with_type_list=entity_or_literal_with_type_list)
        get_hop4(entity_or_literal_with_type_list=entity_or_literal_with_type_list)

    for hop_type, hop_paths in hops.items():
        print(hop_type, hop_paths)
        print('***********************')


if __name__ == '__main__':

    # data_type = 'test' #test dev
    # output_file = oracle_canidates_file + '0102_cwq_9_train_all_questions.json'
    # run_cwq(data_type=data_type, output_file=output_file)

    # data_type = 'train' #test
    # output_file = oracle_canidates_file +'1224_lcquad_train_all_questions.json'
    # run_lcquad(data_type=data_type, output_file=output_file)

    # data_type = 'test' #test
    # output_file = oracle_canidates_file +'1224_graphquestions_test_all_questions_shuang_hop2.json'
    # run_graphquestions(data_type=data_type, output_file=output_file)

    test_one_entity()

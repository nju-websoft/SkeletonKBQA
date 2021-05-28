
from datasets_interface.virtuoso_interface import freebase_kb_interface as kb_interface


if __name__ == '__main__':
    # type.datatime
    # type.float
    # type.int
    # type.enumeration

    # type.datetime
    # literal = '"1844"^^xsd:dateTime'
    # literal = '"1844"^^<http://www.w3.org/2001/XMLSchema#datetime>'
    literal = '"1977-01-09"^^xsd:dateTime'

    # float
    # literal = '"356.72"' #measurement_unit.recurring_money_value.amount
    # literal = '"32"' #measurement_unit.recurring_money_value.amount

    # type.enumeration
    # literal = '"CHE"' #location.country.iso_alpha_3
    # literal = '"250315000417"' #education.school.nces_school_id

    # type.text
    # literal = '"17-14"@en' #{'m.04myq1\tsports.sports_championship_event.result'}
    # literal = '"4 - 0"@en' #sports.sports_championship_event.result

    s_p_set, s_set, p_set = kb_interface.get_s_p_literal_none(literal)
    print(s_p_set)

    # from common.hand_files import read_list_yuanshi
    # literals = read_list_yuanshi('./temp.txt')
    # for literal in literals:
    #     literal_normal = literal_postprocess_cwq(literal)
    #     s_p_set, s_set, p_set = kb_interface.get_s_p_literal_none(literal_normal)
    #     print(literal, '\t', literal_normal, '\t', len(s_p_set))

    print ('end')


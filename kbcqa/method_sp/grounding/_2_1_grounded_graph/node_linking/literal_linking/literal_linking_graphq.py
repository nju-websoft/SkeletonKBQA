from datasets_interface.virtuoso_interface import freebase_kb_interface as kb_interface

if __name__ == '__main__':
    # type.datatime
    # literal = '"1902-07-16"^^<http://www.w3.org/2001/XMLSchema#datetime>'
    # literal = '"1995-04-07"^^<http://www.w3.org/2001/XMLSchema#datetime>'
    # type.float type.int
    # literal = '129.54'
    # literal = '2'
    # literal = '"1768.0"^^<http://www.w3.org/2001/XMLSchema#double>'
    # literal = '"3"^^<http://www.w3.org/2001/XMLSchema#int>'
    literal = '3'
    # literal = '1768.0'
    # literal = '"51"^^<http://www.w3.org/2001/XMLSchema#int>'

    s_p_set, s_set, p_set = kb_interface.get_s_p_literal_none(literal)
    print(s_p_set)

    # type.int
    # ?x1 < 74
    # s_p_set = kb_interface.get_s_p_literal_function('1.524', literal_function='<', literaltype='type.int')
    #?x1 <= 1768.0
    # s_p_set = kb_interface.get_s_p_literal_function('1768.0', literal_function='<=', literaltype='type.float')
    # ?x1 > 1.524
    # s_p_set = kb_interface.get_s_p_literal_function('1.524', literal_function='>', literaltype=None)
    # literal = '"1902-07-16"^^<http://www.w3.org/2001/XMLSchema#datetime>'
    # s_p_set = kb_interface.get_s_p_literal_function(literal, literal_function='>', literaltype='type.datetime')
    # print(s_p_set)

    print('end')

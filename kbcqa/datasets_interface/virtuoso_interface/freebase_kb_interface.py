from common.hand_files import write_set
from datasets_interface.virtuoso_interface.freebase_sparql_odbc import SparqlQueryODBC


sqlodbc = SparqlQueryODBC()


def execute_sparql(sparql):
    query_answers_id_set = sqlodbc.execute_sparql(sparql)
    return query_answers_id_set


def execute_sparql_two_args(sparql):
    sqlodbc.execute_sparql_two_args(sparqlquery=sparql)


def execute_sparql_three_args(sparql):
    sqlodbc.execute_sparql_three_args(sparqlquery=sparql)


def get_s_p_literal_none(literal_value):
    try:
        s_p_set, s_set, p_set = sqlodbc.get_s_p_literal_none(literal_value)
    except Exception as e:
        print('#error:\t', literal_value)
        s_p_set = set()
        s_set = set()
        p_set = set()
    return s_p_set, s_set, p_set


def get_s_p_literal_function(literal_value, literal_function, literaltype):
    s_p = sqlodbc.get_s_p_literal_function(literal_value, literal_function, literaltype)
    return s_p


def get_p_o_by_entity(entity):
    p_o_set, o_set, p1_set = sqlodbc.get_p_o(entity)
    return p_o_set, o_set, p1_set


def get_p_o_literal_by_entity(entity):
    p_o_set, o_set, p1_set = sqlodbc.get_p_o_literal(entity)
    return p_o_set


def get_s_p_by_entity(entity):
    s_p_set, s_set, p2_set = sqlodbc.get_s_p(entity)
    return s_p_set, s_set, p2_set


def get_domain_by_property(property):
    sparql = 'SELECT DISTINCT ?object WHERE { :'+property+' <http://www.w3.org/1999/02/22-rdf-syntax-ns#domain> ?object .}'
    domain_class_set = sqlodbc.execute_sparql(sparql)
    return domain_class_set


def get_range_by_property(property):
    '''object type'''
    sparql = 'SELECT DISTINCT ?object WHERE { :'+property+' <http://www.w3.org/1999/02/22-rdf-syntax-ns#range> ?object .}'
    range_class_set = sqlodbc.execute_sparql(sparql)
    return range_class_set


def get_domain(property_str):
    sparql = 'SELECT DISTINCT ?object WHERE { :'+property_str+' :type.property.schema ?object .}'
    domains_set = sqlodbc.execute_sparql(sparql)
    return domains_set


def get_range(property_str):
    """SELECT DISTINCT ?object WHERE { :'+property_str+' <http://www.w3.org/2000/01/rdf-schema#range> ?object .}"""
    sparql = 'SELECT DISTINCT ?o WHERE { :'+property_str+' :type.property.expected_type ?o}'
    range_set = sqlodbc.execute_sparql(sparql)
    return range_set


def get_p_set(instance_str):
    sparql = 'SELECT DISTINCT ?p WHERE {:'+instance_str+' ?p ?o .}'
    return sqlodbc.execute_sparql(sparql)


def get_classes_of_instance(instance_str):
    """SELECT DISTINCT ?class WHERE {' \
              '{:' + instance_str + ' :type.object.type ?class .}' \
              'UNION ' \
              '{:' + instance_str + ' :common.topic.notable_types ?class}}"""
    sparql = 'SELECT DISTINCT ?class WHERE {:' + instance_str + ' :type.object.type ?class .}'
    classes = sqlodbc.execute_sparql(sparql)
    return classes


def get_all_notable_types():
    '''common.topic.notable_types'''
    sparql = 'SELECT DISTINCT ?type WHERE {?x :common.topic.notable_types ?type. }'
    notable_types = sqlodbc.execute_sparql(sparql)
    for notable_type in notable_types:
        print(notable_type)
    return notable_types


def get_classes_notable_types(instance_str):
    sparql = 'SELECT DISTINCT ?class WHERE {:' + instance_str + ' :common.topic.notable_types ?class}'
    return sqlodbc.execute_sparql(sparql)


def get_all_classes():
    sparql = 'SELECT DISTINCT ?type WHERE {?type :type.object.type  :type.type .}'
    classes = sqlodbc.execute_sparql(sparql)
    human_class_set = set()
    for class_str in classes:
        human_class_set.add(class_str)
    print(len(classes))
    write_set(human_class_set, './mid_class_0121.txt')


def get_names(instance_str):
    sparqlquery = """SELECT DISTINCT ?name  WHERE {
            VALUES ?x0 { :""" + instance_str + """ } . ?x0 :type.object.name ?name . FILTER (langMatches(lang(?name), 'en')).}"""
    results = sqlodbc.execute_sparql(sparqlquery)
    return results


def get_alias(instance_str):
    sparqlquery = """SELECT DISTINCT ?name  WHERE {
            VALUES ?x0 { :""" + instance_str + """ } . ?x0 :common.topic.alias ?name . FILTER (langMatches(lang(?name), 'en')).}"""
    results = sqlodbc.execute_sparql(sparqlquery)
    return results


def get_all_instances():
    sparql = "SELECT DISTINCT ?instance ?name WHERE{" \
             "?instance :type.object.type ?type." \
             "?type :type.object.type :type.type." \
             "?instance :type.object.name ?name. " \
             "FILTER(langMatches(lang(?name), 'en')).}"
    instances = sqlodbc.execute_sparql_two_args(sparql)


def get_s_count_by_property(property_str):
    sparql = 'SELECT count(DISTINCT ?s) WHERE {?s :'+property_str+' ?o .}'
    answers = sqlodbc.execute_sparql(sparql)
    return answers.pop()


def get_s_o_by_property(property_str):
    sparql = 'SELECT DISTINCT ?s ?o WHERE {?s :'+property_str+' ?o .}'
    answers = sqlodbc.execute_sparql_two_args(sparql)
    return answers


def get_instance_properties_by_class(class_str):
    sparql = 'SELECT DISTINCT ?o WHERE {:'+class_str+' :type.type.instance ?o .}'
    instances = sqlodbc.execute_sparql(sparql)
    related_properties_set = set()
    for instance in instances:
        related_properties_set.add(get_p_set(instance_str=instance))
    return related_properties_set


def get_instance_by_class(class_str):
    sparql = 'SELECT count(DISTINCT ?instance) WHERE {?instance :type.object.type :'+class_str+'.}'
    instances_set = sqlodbc.execute_sparql(sparql)
    return instances_set


def get_instance_by_class_notable_type(class_str):
    sparql = 'SELECT DISTINCT ?instance WHERE {?instance :common.topic.notable_types :'+class_str+'}'
    return sqlodbc.execute_sparql(sparql)


def get_all_properties():
    sparql = 'SELECT DISTINCT ?relation WHERE { ?relation :type.object.type  :type.property .}'
    properties = sqlodbc.execute_sparql(sparql)
    for property in properties:
        print (property)


def get_all_properties_with_count():
    sparql = 'SELECT DISTINCT ?relation WHERE { ?relation :type.object.type  :type.property .}'
    properties = sqlodbc.execute_sparql(sparql)
    properties_with_count_list = []
    error_property_list = []
    for property in properties:
        try:
            instances_count = get_s_o_by_property(property)
            print(property, instances_count)
            properties_with_count_list.append(('%s\t%d') % (property, instances_count))
        except Exception as e:
            print('#error!!!', property)
            error_property_list.append(property)
    print(len(properties))
    write_set(properties_with_count_list, './properties_with_count.txt')
    print(error_property_list)


def get_all_reverse_properties():
    sparql = 'PREFIX : <http://rdf.freebase.com/ns/> SELECT ?s ?o WHERE { ?s :type.property.reverse_property ?o}'
    execute_sparql_two_args(sparql)


def get_numerical_properties():
    numerical_property_tuple_list = []
    #type datatime
    sparql = 'PREFIX : <http://rdf.freebase.com/ns/> SELECT distinct ?p WHERE { ?p :type.property.expected_type :type.datetime.}'
    properties = execute_sparql(sparql)
    for property in properties:
        numerical_property_tuple_list.append((property, 'type.datatime'))

    #type float
    sparql = 'PREFIX : <http://rdf.freebase.com/ns/> SELECT distinct ?p WHERE { ?p :type.property.expected_type :type.float.}'
    properties = execute_sparql(sparql)
    for property in properties:
        numerical_property_tuple_list.append((property, 'type.float'))

    #type int
    sparql = 'PREFIX : <http://rdf.freebase.com/ns/> SELECT distinct ?p WHERE { ?p :type.property.expected_type :type.int.}'
    properties = execute_sparql(sparql)
    for property in properties:
        numerical_property_tuple_list.append((property, 'type.int'))

    #type.enumeration
    sparql = 'PREFIX : <http://rdf.freebase.com/ns/> SELECT distinct ?p WHERE { ?p :type.property.expected_type :type.enumeration.}'
    properties = execute_sparql(sparql)
    for property in properties:
        numerical_property_tuple_list.append((property, 'type.enumeration'))

    #type.text
    sparql = 'PREFIX : <http://rdf.freebase.com/ns/> SELECT distinct ?p WHERE { ?p :type.property.expected_type :type.text.}'
    properties = execute_sparql(sparql)
    for property in properties:
        numerical_property_tuple_list.append((property, 'type.text'))
    for i, (property, property_expected_type) in enumerate(numerical_property_tuple_list):
        try:
            sparql = '''SELECT count(?s) WHERE { ?s :''' + property + ''' ?o }'''
            count = execute_sparql(sparql)
            if count.pop() == 0:
                continue
        except Exception as e:
            continue
        print(('%s\t%s') % (property, property_expected_type))
    print(len(numerical_property_tuple_list))


def get_quotation_instance():
    sparql = 'PREFIX : <http://rdf.freebase.com/ns/> SELECT distinct ?s WHERE { ?s :type.object.type :media_common.quotation.}'
    instances = execute_sparql(sparql)
    for instance in instances:
        names = get_names(instance)
        if len(names) > 0:
            print (('%s\t%s') % (instance, names.pop()))


if __name__ == "__main__":
    pass
    # names = get_names('m.019v9k')
    get_numerical_properties()
    # kb_interface.get_all_reverse_properties()
    # get_all_classes()
    # print (name.encode('utf-8').decode('latin1'))
    # get_properties()

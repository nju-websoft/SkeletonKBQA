from datasets_interface.virtuoso_interface.dbpedia_sparql_odbc import SparqlQueryODBC

sqlodbc = SparqlQueryODBC()


version = '<http://dbpedia201604.org>'


#获得两个实体之间的所有triples
def get_all_triples(entity_a, entity_b):
    triples = set()
    entity_a_s_p, entity_a_p_o = get_s_p_p_o_bylinkedentity(entity_a)
    for s_p in entity_a_s_p:
        s, p = s_p.split('\t')
        if s == entity_b:
            triples.add(entity_b + '\t' + p + '\t' + entity_a)
    for p_o in entity_a_p_o:
        p, o = p_o.split('\t')
        if o == entity_b:
            triples.add(entity_a + '\t' + p + '\t' + entity_b)
    entity_b_s_p, entity_b_p_o = get_s_p_p_o_bylinkedentity(entity_b)
    for s_p in entity_b_s_p:
        s, p = s_p.split('\t')
        if s == entity_a:
            triples.add(entity_a + '\t' + p + '\t' + entity_b)
    for p_o in entity_b_p_o:
        p, o = p_o.split('\t')
        if o == entity_a:
            triples.add(entity_b + '\t' + p + '\t' + entity_a)
    return triples


# 读取1步路径
def get_s_p_p_o_bylinkedentity(entity):
    '''获得entity两个方向一步path,得到entity的s_p, p_o'''
    p_o, o_set, p_set = get_p_o_by_entity(entity)
    s_p, s_set, p_set = get_s_p_by_entity(entity)
    return s_p, p_o


def get_p_o_by_entity(entity):
    p_o_set, o_set, p1_set = sqlodbc.get_p_o(entity)
    return p_o_set, o_set, p1_set


def get_s_p_by_entity(entity):
    s_p_set, s_set, p2_set = sqlodbc.get_s_p(entity)
    return s_p_set, s_set, p2_set


def execute_sparql_exist(sparqlquery):
    answers = sqlodbc.execute_sparql_one_args(sparqlquery=sparqlquery)
    print(answers)


def execute_sparql(sparql):
    try:
        query_answers_id_set = sqlodbc.execute_sparql_one_args(sparql)
    except Exception as e:
        query_answers_id_set = set()
    return query_answers_id_set


def get_all_resources_with_labels():
    sparqlquery = 'SELECT DISTINCT ?s ?label from '+version+' WHERE {' \
                ' ?s ?p ?o. ?s rdfs:label ?label. FILTER(langMatches(lang(?label), \'en\')). ' \
                'Filter isIRI(?s) }'
    answers_set = execute_sparql(sparql=sparqlquery)
    return answers_set


def get_all_resources():
    sparqlquery = 'SELECT DISTINCT ?s from '+version+' WHERE { ?s ?p ?o. Filter isIRI(?s)}'
    answers_set = execute_sparql(sparql=sparqlquery)
    return answers_set


def get_all_classes():
    sparqlquery = 'SELECT DISTINCT ?s ?label from '+version+' WHERE {' \
                  ' ?s rdf:type owl:Class. ?s rdfs:label ?label. FILTER(langMatches(lang(?label), \'en\')). }'
    answers = execute_sparql(sparql=sparqlquery)
    return answers


def get_all_wikiPageRedirects():
    sparqlquery = 'SELECT DISTINCT ?s ?slabel ?o from '+version+' WHERE {' \
                  ' ?s dbo:wikiPageRedirects ?o. ?s rdfs:label ?slabel.' \
                  'Filter(langMatches(lang(?slabel), \'en\'))}'
    answers = execute_sparql(sparql=sparqlquery)


def get_wikiPageRedirects(resource):
    sparqlquery = 'SELECT DISTINCT ?o from '+version+' WHERE { <'+resource+'> dbo:wikiPageRedirects ?o.}'
    return execute_sparql(sparql=sparqlquery)


def get_all_linkText():
    sparqlquery = 'select distinct ?s ?label from '+version+' where { ?s dbo:wikiPageWikiLinkText ?label. Filter(langMatches(lang(?label), \'en\'))}'
    answers = execute_sparql(sparql=sparqlquery)
    return answers


def get_label(resource):
    sparqlquery = 'SELECT DISTINCT ?label from '+version+' WHERE {' \
                  ' <'+resource+'> rdfs:label ?label.' \
                  ' FILTER(langMatches(lang(?label), \'en\')).}'
    answers = execute_sparql(sparql=sparqlquery)
    return answers


def get_out_edge_degree(resource):
    sparqlquery = 'SELECT count(?p) from '+version+' WHERE { ' \
                '{<'+resource+'> ?p ?o.} union { ?s ?p <'+resource+'> .}' \
                '}'
    answers = execute_sparql(sparql=sparqlquery)
    return answers


def get_class_of_resource(resource):
    '''SELECT DISTINCT ?type WHERE { res:Christopherson_Business_Travel rdf:type ?type }'''
    sparqlquery = 'SELECT DISTINCT ?uri from '+version+' WHERE { <'+resource+'> rdf:type ?uri }'
    return execute_sparql(sparql=sparqlquery)


def get_instances_of_class(class_):
    '''?uri rdf:type dbo:Mountain class_ = '<http://dbpedia.org/ontology/Bird>'''
    sparqlquery = 'SELECT DISTINCT ?resource from '+version+' WHERE { ?resource rdf:type '+class_+' }'
    return execute_sparql(sparql=sparqlquery)


def get_domain_of_property(property):
    sparqlquery = 'SELECT DISTINCT ?o from '+version+' WHERE { '+property+' rdfs:domain ?o }'
    answers = execute_sparql(sparql=sparqlquery)
    return answers


def get_range_of_property(property):
    sparqlquery = 'SELECT DISTINCT ?o from ' + version + ' WHERE { '+property+' rdfs:range ?o }'
    answers = execute_sparql(sparql=sparqlquery)
    return answers


def get_dbo_datatype_property():
    sparqlquery = 'SELECT DISTINCT ?s ?label from '+version+' WHERE {' \
                  '?s rdf:type owl:DatatypeProperty. ?s rdfs:label ?label. FILTER(langMatches(lang(?label), \'en\')).}'
    answers = execute_sparql(sparql=sparqlquery)
    return answers


def get_dbo_object_property():
    sparqlquery = 'SELECT DISTINCT ?s ?label from '+version+' WHERE {' \
                  '?s rdf:type owl:ObjectProperty. ?s rdfs:label ?label. FILTER(langMatches(lang(?label), \'en\')).}'
    answers = execute_sparql(sparql=sparqlquery)
    return answers


def get_dbp_property():
    sparqlquery = 'SELECT DISTINCT ?s WHERE { ?s rdf:type rdf:Property }'
    answers = execute_sparql(sparql=sparqlquery)
    # filter dbo property
    dbp_property_list = []
    error_property_list = []
    for answer in answers:
        if 'http://dbpedia.org/property/' in answer:
            dbp_property_list.append(answer)
            try:
                label_set = get_label(answer)
                print(('%s\t%s') % (answer, str(label_set)))
            except Exception as e:
                error_property_list.append(answer)
    print('#error:\t', error_property_list)
    return dbp_property_list


if __name__ == "__main__":
    # sparqlquery = 'SELECT DISTINCT ?uri WHERE { ?uri rdf:type dbo:Ship ; dct:subject dbc:Christopher_Columbus ; dct:subject dbc:Exploration_ships }'
    # get_all_resources()
    # get_all_resources_with_labels()
    # get_all_classes()
    # get_dbo_datatype_property()
    # get_dbo_object_property()
    # get_all_wikiPageRedirects()
    # get_all_linkText()
    # get_dbp_property()
    # resource = '<http://dbpedia.org/resource/Arthropod>'
    # resource = 'res:Cerro_Moneda'
    # p_o_set, o_set, p_set = sqlodbc.get_p_o(resource)
    # for p_o in p_o_set:
    #     print (p_o)
    # degree = get_out_edge_degree('http://dbpedia.org/resource/Salt_lake_city')
    # print (degree)
    # s_p, p_o = get_s_p_p_o_bylinkedentity('http://dbpedia.org/resource/Pizza')
    # execute_sparql_exist(sparqlquery=sparqlquery)
    types = get_class_of_resource('http://dbpedia.org/resource/Juan_Carlos_Valer\u00f3n')
    print (types)

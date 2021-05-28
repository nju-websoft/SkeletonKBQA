from SPARQLWrapper import JSON, SPARQLWrapper
from common.globals_args import argument_parser
from method_sp.grounding.grounding_args import kb_relations


class SparqlQueryHTML():

    def __init__(self):
        self.dbpedia_sparql = SPARQLWrapper(argument_parser.dbpedia_sparql_html_info)
        self.dbpedia_prefix = 'http://dbpedia.org/'
        self.resource_prefix = 'http://dbpedia.org/resource/'

    def return_str_not_something(self, variable):
        '''filter'''
        return "FILTER ("+variable+"!= <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>) ." \
               "FILTER ("+variable+"!= <http://xmlns.com/foaf/0.1/name>) ." \
               "FILTER ("+variable+"!= <http://www.w3.org/2000/01/rdf-schema#label>) ." \
               "FILTER ("+variable+"!= <http://www.w3.org/2000/01/rdf-schema#seeAlso>) ." \
               "FILTER ("+variable+"!= <http://xmlns.com/foaf/0.1/nick>) ." \
               "FILTER ("+variable+"!= <http://dbpedia.org/ontology/wikiPageRedirects>)." \
               "FILTER ("+variable+"!= <http://dbpedia.org/ontology/wikiPageWikiLinkText>). " \
               "FILTER ("+variable+"!= <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>). "

    def filter_relation(self, relation):
        '''http://dbpedia.org/property/'''
        if self.dbpedia_prefix not in relation:
            return False
        if relation not in kb_relations:
            return False
        return relation

    def filter_entity(self, entity):
        return entity

    def get_p_o(self, s):
        '''获取s, 出边信息
            :return p_o_set, o_set, p_set
        '''
        p_o_set = set()
        o_set = set()
        p_set = set()
        sparqlquery = """SELECT DISTINCT ?p ?o  WHERE {  """ + self.return_str_not_something("?p") + \
                      """{<""" +self.resource_prefix+ s + """> ?p ?o . }}""" # limit 10
        self.dbpedia_sparql.setQuery(sparqlquery)
        self.dbpedia_sparql.setReturnFormat(JSON)
        try:
            results = self.dbpedia_sparql.query().convert()
            for result in results["results"]["bindings"]:
                p = result['p']['value']
                o = result['o']['value']
                p_o_set.add("\t".join([p, o]))
                p_set.add(p)
                o_set.add(o)
        except Exception:
            print('Error!')
        return p_o_set, o_set, p_set

    def get_s_p(self, o):
        '''获得o 的入边信息
         :return s_p_set, s_set, p_set'''
        s_p_set = set()
        s_set = set()
        p_set = set()
        sparqlquery = """SELECT DISTINCT ?s ?p  WHERE {  """ + self.return_str_not_something("?p") + \
                      """{?s ?p <""" +self.resource_prefix+ o + """>. }}""" # limit 10
        self.dbpedia_sparql.setQuery(sparqlquery)
        self.dbpedia_sparql.setReturnFormat(JSON)
        try:
            results = self.dbpedia_sparql.query().convert()
            for result in results["results"]["bindings"]:
                s = result['s']['value']
                p = result['p']['value']
                s_p_set.add("\t".join([s, p]))
                s_set.add(s)
                p_set.add(p)
        except Exception as e:
            print('Error!', e)
        return s_p_set, s_set, p_set

    def get_s_p_literal_none(self, literal):
        '''读取literal的入边信息
        :return s_p_set
        '''
        s_p_set = set()
        s_set = set()
        p_set = set()
        sparqlquery = """ SELECT DISTINCT ?s ?p  WHERE { VALUES ?x1 { <""" + literal + """> } . ?s ?p ?x1 . } """
        self.dbpedia_sparql.setQuery(sparqlquery)
        self.dbpedia_sparql.setReturnFormat(JSON)
        try:
            results = self.dbpedia_sparql.query().convert()
            for result in results["results"]["bindings"]:
                s = result['s']['value']
                p = result['p']['value']
                s = self.filter_entity(s)
                p = self.filter_relation(p)
                if s and p:
                    s_p_set.add("\t".join([s, p]))
                    s_set.add(s)
                    p_set.add(p)
        except Exception as e:
            print('Error!', e)
        return s_p_set, s_set, p_set

    def get_p1_p2_by_entity(self, s, label):
        p1_p2_set = set()
        sparql_query = """ SELECT DISTINCT ?p1,?p2  WHERE { """ + self.return_str_not_something("?p1") + self.return_str_not_something("?p2")
        if label[0] == '+':
            sparql_query += """ <"""+self.resource_prefix+s+"""> ?p1  ?o1 . """
        else:
            sparql_query += """ ?o1 ?p1  <"""+self.resource_prefix+s+"""> . """
        if label[1] == '+':
            sparql_query += """ ?o1 ?p2 ?o2 . }"""
        else:
            sparql_query += """ ?o2 ?p2 ?o1 . }"""
        self.dbpedia_sparql.setQuery(sparql_query)
        self.dbpedia_sparql.setReturnFormat(JSON)
        try:
            results = self.dbpedia_sparql.query().convert()
            for result in results["results"]["bindings"]:
                p1 = result['p1']['value']
                p2 = result['p2']['value']
                p1 = self.filter_relation(p1)
                p2 = self.filter_relation(p2)
                if p1 and p2:
                    p1_p2_set.add("\t".join([p1, p2]))
        except Exception as e:
            print('Error!', e)
        return p1_p2_set

    def get_p1_p2_by_literal(self, literal_value, label):
        '''"1995-04-07"^^<http://www.w3.org/2001/XMLSchema#datetime>'''
        p1_p2_set = set()
        sparql_query = """ SELECT DISTINCT  ?p1,?p2  WHERE { """ + self.return_str_not_something("?p1") + self.return_str_not_something("?p2")
        if label[0] == '+':
            return p1_p2_set  # do not chu edge
        else:
            sparql_query += """  ?o1 ?p1  <""" + literal_value + """>  . """
        if label[1] == '+':
            sparql_query += """ ?o1 ?p2 ?o2 .  }"""
        else:
            sparql_query += """ ?o2 ?p2 ?o1 .  }"""
        self.dbpedia_sparql.setQuery(sparql_query)
        self.dbpedia_sparql.setReturnFormat(JSON)
        try:
            results = self.dbpedia_sparql.query().convert()
            for result in results["results"]["bindings"]:
                p1 = result['p1']['value']
                p2 = result['p2']['value']
                p1 = self.filter_relation(p1)
                p2 = self.filter_relation(p2)
                if p1 and p2:
                    p1_p2_set.add("\t".join([p1, p2]))
        except Exception as e:
            print('Error!', e)
        return p1_p2_set

    def get_p1_p2_by_e1_e2(self, e1, e2, label):
        '''
        :param e1:
        :param e2:
        :param label:
        :return: p1_p2_set
        '''
        p1_p2_set = set()
        sparql_query = """ SELECT DISTINCT  ?p1,?p2  WHERE { """ + self.return_str_not_something("?p1") + self.return_str_not_something("?p2")
        if label[0] == '+':
            sparql_query += """ { <"""+self.resource_prefix + e1 + """> ?p1  ?o1 . """
        else:
            sparql_query += """ { ?o1 ?p1 <"""+self.resource_prefix + e1 + """>  . """

        if label[1] == '+':
            sparql_query += """ <"""+self.resource_prefix+e2+"""> ?p2 ?o1 . }}"""
        else:
            sparql_query += """ ?o1 ?p2 <"""+self.resource_prefix+e2+"""> . }}"""

        self.dbpedia_sparql.setQuery(sparql_query)
        self.dbpedia_sparql.setReturnFormat(JSON)
        try:
            results = self.dbpedia_sparql.query().convert()
            for result in results["results"]["bindings"]:
                p1 = result['p1']['value']
                p2 = result['p2']['value']
                p1 = self.filter_relation(p1)
                p2 = self.filter_relation(p2)
                if p1 and p2:
                    p1_p2_set.add("\t".join([p1, p2]))
        except Exception as e:
            print('Error!', e)
        return p1_p2_set

    def get_p1_by_e1_e2(self, e1, e2, label):
        p1_set = set()
        sparql_query = """ SELECT DISTINCT ?p1  WHERE { """ + self.return_str_not_something("?p1")
        if label[0] == '+':
            sparql_query += """ <""" + self.resource_prefix + e1 + """> ?p1 <""" + self.resource_prefix + e2 + """> . }"""
        else:
            sparql_query += """ <"""+self.resource_prefix + e2 + """> ?p1 <""" + self.resource_prefix + e1 + """> . }"""

        self.dbpedia_sparql.setQuery(sparql_query)
        self.dbpedia_sparql.setReturnFormat(JSON)
        try:
            results = self.dbpedia_sparql.query().convert()
            for result in results["results"]["bindings"]:
                p1 = result['p1']['value']
                p1 = self.filter_relation(p1)
                if p1:
                    p1_set.add(p1)
        except Exception as e:
            print('Error!', e)
        return p1_set

    def execute_sparql_one_args(self,sparqlquery):
        '''{'head': {'link': [], 'vars': ['uri']}, 'results': {'distinct': False, 'ordered': True,
        'bindings': [{'uri': {'type': 'uri', 'value': 'http://dbpedia.org/resource/Upper_Neretva'}}]}}'''
        self.dbpedia_sparql.setQuery(sparqlquery)
        self.dbpedia_sparql.setReturnFormat(JSON)
        answers = set()
        try:
            results = self.dbpedia_sparql.query().convert()
            for result in results["results"]["bindings"]:
                answers.add(result['uri']["value"])
        except Exception as e:
            print ('Error!', e)
        return answers

    def execute_sparql(self,sparqlquery):
        self.dbpedia_sparql.setQuery(sparqlquery)
        self.dbpedia_sparql.setReturnFormat(JSON)
        try:
            results = self.dbpedia_sparql.query().convert()
        except Exception:
            return set()
        answers=set()
        # print(results)
        if 'boolean' in results:

            answers.add(results["boolean"])
            return answers
        if len(results["results"]["bindings"])==1:
            result=results["results"]["bindings"][0]

            if 'callret-0' in result:
                if result["callret-0"]["type"]=="typed-literal":
                    answers.add(result["callret-0"]["value"])
                    return answers

        for result in results["results"]["bindings"]:
            answers.add(self.filter_entity(result["uri"]["value"]))

        return answers

    def get_p1_by1entity(self, entity,label):
        '''获得o 的入边信息
         :return s_p_set, s_set, p_set'''
        p_set=set()
        sparqlquery = """SELECT DISTINCT ?p  WHERE {  """ + self.return_str_not_something("?p")
        if label=='-':
            sparqlquery+="""{?s ?p <""" +entity + """>. }}"""
        else:
            sparqlquery+="""{<""" + entity + """> ?p ?o . }} """
        self.dbpedia_sparql.setQuery(sparqlquery)
        self.dbpedia_sparql.setReturnFormat(JSON)
        try:
            results = self.dbpedia_sparql.query().convert()
            for result in results["results"]["bindings"]:
                p = result['p']['value']

                if self.filter_relation(p):
                    p_set.add(p)
        except Exception as e:
            print('Error!', e)
        return p_set

    def get_p1_p2_by_1entity(self, entity, label):
        p1_p2_set = set()
        sparql_query = """ SELECT DISTINCT ?p1,?p2  WHERE { """ + self.return_str_not_something("?p1") + self.return_str_not_something("?p2")
        if label[0] == '+':
            sparql_query += """ <"""+entity+"""> ?p1  ?o1 . """
        else:
            sparql_query += """ ?o1 ?p1  <"""+entity+"""> . """
        if label[1] == '+':
            sparql_query += """ ?o1 ?p2 ?o2 . }"""
        else:
            sparql_query += """ ?o2 ?p2 ?o1 . }"""
        self.dbpedia_sparql.setQuery(sparql_query)
        self.dbpedia_sparql.setReturnFormat(JSON)
        try:
            results = self.dbpedia_sparql.query().convert()
            for result in results["results"]["bindings"]:
                p1 = result['p1']['value']
                p2 = result['p2']['value']
                p1 = self.filter_relation(p1)
                p2 = self.filter_relation(p2)
                if p1 and p2:
                    p1_p2_set.add("\t".join([p1, p2]))
        except Exception as e:
            print('Error!', e)
        return p1_p2_set

    def get_p1_p2_by_2entity(self, e1, e2, label):
        '''
        :param e1:
        :param e2:
        :param label:
        :return: p1_p2_set
        '''
        p1_p2_set = set()
        sparql_query = """ SELECT DISTINCT  ?p1,?p2  WHERE { """ + self.return_str_not_something(
            "?p1") + self.return_str_not_something("?p2")
        if label[0] == '+':
            sparql_query += """ { <""" +e1 + """> ?p1  ?o1 . """
        else:
            sparql_query += """ { ?o1 ?p1 <""" + e1 + """>  . """

        if label[1] == '+':
            sparql_query += """ <""" + e2 + """> ?p2 ?o1 . }}"""
        else:
            sparql_query += """ ?o1 ?p2 <""" +  e2 + """> . }}"""

        self.dbpedia_sparql.setQuery(sparql_query)
        self.dbpedia_sparql.setReturnFormat(JSON)
        try:
            results = self.dbpedia_sparql.query().convert()
            for result in results["results"]["bindings"]:
                p1 = result['p1']['value']
                p2 = result['p2']['value']
                p1 = self.filter_relation(p1)
                p2 = self.filter_relation(p2)
                if p1 and p2:
                    p1_p2_set.add("\t".join([p1, p2]))
        except Exception as e:
            print('Error!', e)
        return p1_p2_set

    def get_p1_by_2enttity(self, e1, e2, label):
        p1_set = set()
        sparql_query = """ SELECT DISTINCT ?p1  WHERE { """ + self.return_str_not_something("?p1")
        if label[0] == '+':
            sparql_query += """ <""" + e1 + """> ?p1 <""" + e2 + """> . }"""
        else:
            sparql_query += """ <""" +  e2 + """> ?p1 <"""  + e1 + """> . }"""

        self.dbpedia_sparql.setQuery(sparql_query)
        self.dbpedia_sparql.setReturnFormat(JSON)
        # print(sparql_query)
        try:
            results = self.dbpedia_sparql.query().convert()
            for result in results["results"]["bindings"]:
                p1 = result['p1']['value']
                # print(p1)
                p1 = self.filter_relation(p1)
                if p1:
                    p1_set.add(p1)
        except Exception as e:
            print('Error!', e)
        return p1_set


if __name__ == '__main__':
    sqlodbc = SparqlQueryHTML()
    """
    sparqlquery = 'SELECT DISTINCT ?p ?o WHERE { <http://dbpedia.org/resource/Michał_Vituška> ?p ?o. }'
    sparqlquery = 'PREFIX : <http://dbpedia.org/resource/> SELECT DISTINCT ?uri WHERE { VALUES ?x1 { :1906\u201317_Stanford_rugby_teams } . ?x1 <http://dbpedia.org/property/year> ?uri .}'
    answers = sqlodbc.execute_sparql_one_args(sparqlquery=sparqlquery)
    print(answers)
    """
    uri = '1906\u201317_Stanford_rugby_teams'
    p_o_set, o_set, p_set = sqlodbc.get_p_o(s=uri)
    print(p_set)


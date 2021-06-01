#__*__ encoding: utf-8 __*__
import pyodbc
from common.globals_args import argument_parser
from method_sp.grounding.grounding_args import kb_relations


class SparqlQueryODBC():

    def __init__(self):
        self.dbpedia_sparql = pyodbc.connect(argument_parser.dbpedia_pyodbc, ansi=True, autocommit=True, timeout=500000)
        self.dbpedia_sparql.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
        self.dbpedia_sparql.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
        # self.dbpedia_sparql.setencoding(encoding='utf8')
        self.prefix = "sparql PREFIX : <http://dbpedia.org/resource/> " \
                      "PREFIX dbp: <http://dbpedia.org/property/>" \
                      "PREFIX dbo: <http://dbpedia.org/ontology/>" \
                      "PREFIX dct: <http://purl.org/dc/terms/> " \
                      "PREFIX dbc: <http://dbpedia.org/resource/Category:>" \
                      "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>" \
                      "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>" \
                      "PREFIX dbr: <http://dbpedia.org/resource/>"
        self.dbpedia_prefix = 'http://dbpedia.org/'
        self.resource_prefix = 'http://dbpedia.org/resource/'

    def return_str_not_something(self, variable):
        '''filter
        predicate blacklist
        <http://www.w3.org/2000/01/rdf-schema#seeAlso>
        <http://purl.org/linguistics/gold/hypernym>
        <http://www.w3.org/2000/01/rdf-schema#label>
        <http://www.w3.org/2000/01/rdf-schema#comment>
        <http://purl.org/voc/vrank#hasRank>
        <http://xmlns.com/foaf/0.1/isPrimaryTopicOf>
        <http://xmlns.com/foaf/0.1/primaryTopic>
        <http://dbpedia.org/ontology/abstract>
        <http://dbpedia.org/ontology/thumbnail>
        <http://dbpedia.org/ontology/wikiPageExternalLink>
        <http://dbpedia.org/ontology/wikiPageRevisionID>
        <http://dbpedia.org/ontology/type>
        <http://dbpedia.org/ontology/wikiPageWikiLink>
        <http://dbpedia.org/ontology/wikiPageID>
        <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
        <http://xmlns.com/foaf/0.1/primaryTopic>
        <http://dbpedia.org/ontology/wikiPageDisambiguates>
        <http://dbpedia.org/ontology/wikiPageRedirects>
        <http://www.w3.org/ns/prov#wasDerivedFrom>
        <http://dbpedia.org/ontology/wikiPageLength>
        <http://xmlns.com/foaf/0.1/depiction>
        <http://dbpedia.org/property/id>
        <http://xmlns.com/foaf/0.1/homepage>
        <http://dbpedia.org/ontology/wikiPageOutDegree>
        <http://xmlns.com/foaf/0.1/name>
        <http://www.w3.org/2004/02/skos/core#broader>
        <http://www.w3.org/2004/02/skos/core#prefLabel>
        <http://purl.org/dc/terms/subject>
        '''
        return "FILTER ("+variable+"!= rdf:type) . " \
               "FILTER ("+variable+"!= <http://xmlns.com/foaf/0.1/name>) . " \
               "FILTER ("+variable+"!= rdfs:label) . " \
               "FILTER ("+variable+"!= rdfs:seeAlso) . " \
               "FILTER ("+variable+"!= <http://xmlns.com/foaf/0.1/nick>) . " \
               "FILTER ("+variable+"!= <http://dbpedia.org/ontology/wikiPageRedirects>). " \
               "FILTER ("+variable+"!= <http://dbpedia.org/ontology/wikiPageWikiLinkText>). "

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
        sparqlquery = """SELECT DISTINCT ?p ?o  WHERE {  """ + self.return_str_not_something("?p") +\
                      """ <""" + self.resource_prefix + s + """> ?p ?o . }"""
        cursor = self.dbpedia_sparql.cursor()
        results = cursor.execute(self.prefix + sparqlquery)
        for result in results:
            p = result[0]
            o = result[1]
            p = self.filter_relation(p)
            o = self.filter_entity(o)
            if p and o:
                p_o_set.add("\t".join([p, o]))
                p_set.add(p)
                o_set.add(o)
        return p_o_set, o_set, p_set

    def get_s_p(self, o):
        '''获得o 的入边信息
         :return s_p_set, s_set, p_set'''
        s_p_set = set()
        s_set = set()
        p_set = set()
        sparqlquery = """SELECT DISTINCT ?s ?p  WHERE {  """ + self.return_str_not_something("?p") +\
                      """ ?s ?p <"""+self.resource_prefix+o+"""> . }"""
        cursor = self.dbpedia_sparql.cursor()
        results = cursor.execute(self.prefix + sparqlquery)
        for result in results:
            s = result[0]
            p = result[1]
            s_p_set.add("\t".join([s, p]))
            s_set.add(s)
            p_set.add(p)
        return s_p_set, s_set, p_set

    def get_s_p_literal_none(self, literal):
        '''读取literal的入边信息
        :return s_p_set
        '''
        s_p_set = set()
        s_set = set()
        p_set = set()
        results = self.dbpedia_sparql.execute(
            self.prefix + """ SELECT DISTINCT ?s ?p  WHERE { VALUES ?x1 { <""" + literal + """> } . ?s ?p ?x1 . } """)  # limit 100000
        i = 0
        j = 0
        for result in results:
            if i < 10000:
                i += 1
            else:
                j += 1
                i = 0
                # print(time.strftime('%Y.%m.%d %H:%M:%S ', time.localtime(time.time())))
            s = result[0]
            p = result[1]
            s = self.filter_entity(s)
            p = self.filter_relation(p)
            if s and p:
                s_p_set.add("\t".join([s, p]))
                s_set.add(s)
                p_set.add(p)
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
        results = self.dbpedia_sparql.execute(self.prefix + sparql_query)
        for i, result in enumerate(results):
            p1 = result[0]
            p2 = result[1]
            p1 = self.filter_relation(p1)
            p2 = self.filter_relation(p2)
            if p1 and p2:
                p1_p2_set.add("\t".join([p1, p2]))
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
        results = self.dbpedia_sparql.execute(self.prefix + sparql_query)
        for result in results:
            p1 = result[0]
            p2 = result[1]
            p1 = self.filter_relation(p1)
            p2 = self.filter_relation(p2)
            if p1 and p2:
                p1_p2_set.add("\t".join([p1, p2]))
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
        results = self.dbpedia_sparql.execute(self.prefix + sparql_query)
        for result in results:
            p1 = result[0]
            p2 = result[1]
            p1 = self.filter_relation(p1)
            p2 = self.filter_relation(p2)
            if p1 and p2:
                p1_p2_set.add("\t".join([p1, p2]))
        return p1_p2_set

    def get_p1_by_e1_e2(self, e1, e2, label):
        p1_set = set()
        sparql_query = """ SELECT DISTINCT ?p1  WHERE { """ + self.return_str_not_something("?p1")
        if label[0] == '+':
            sparql_query += """ <""" + self.resource_prefix + e1 + """> ?p1 <""" + self.resource_prefix + e2 + """> . }"""
        else:
            sparql_query += """ <"""+self.resource_prefix + e2 + """> ?p1 <""" + self.resource_prefix + e1 + """> . }"""

        results = self.dbpedia_sparql.execute(self.prefix + sparql_query)
        for result in results:
            p1 = result[0]
            p1 = self.filter_relation(p1)
            if p1:
                p1_set.add(p1)
        return p1_set

    def execute_sparql_one_args(self, sparqlquery):
        cursor = self.dbpedia_sparql.cursor()
        results = cursor.execute(self.prefix+sparqlquery)
        answers = set()
        for result in results:
            if isinstance(result[0], str) and self.dbpedia_prefix in result[0] and False:
                answers.add(result[0].replace(self.dbpedia_prefix, ""))
            else:
                answers.add(result[0])
        return answers

    def execute_sparql_two_args(self, sparqlquery):
        '''return two args'''
        cursor = self.dbpedia_sparql.cursor()
        results = cursor.execute(self.prefix+sparqlquery)
        for result in results:
            instance = result[0]
            if isinstance(instance, str) and self.dbpedia_prefix in instance:
                instance = instance.replace(self.dbpedia_prefix, "")
            class_str = result[1]
            if isinstance(class_str, str) and self.dbpedia_prefix in class_str:
                class_str = class_str.replace(self.dbpedia_prefix, "")
            print(('%s\t%s')%(instance, class_str))

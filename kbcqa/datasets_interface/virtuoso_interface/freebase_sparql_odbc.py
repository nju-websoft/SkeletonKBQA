import pyodbc
import time

from method_sp.grounding.grounding_args import kb_relations
from common.globals_args import argument_parser


class SparqlQueryODBC():

    def __init__(self):
        self.freebase_sparql = pyodbc.connect(argument_parser.freebase_pyodbc, ansi=True, autocommit=True, timeout=500000)
        self.freebase_sparql.setdecoding(pyodbc.SQL_CHAR, encoding='utf8')
        self.freebase_sparql.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
        self.freebase_sparql.setencoding(encoding='utf8')
        # self.freebase_sparql.timeout = 1
        self.freebase_prefix = "http://rdf.freebase.com/ns/"
        self.prefix = "sparql PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> " \
                      "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
                      "PREFIX : <http://rdf.freebase.com/ns/> "

    def return_str_not_something(self,variable):
        '''filter'''
        return "FILTER ("+variable+"!= :type.object.type) . " \
               "FILTER ("+variable+"!= :common.topic.notable_types) ." \
               "FILTER ("+variable+" != rdf:type) ."

    def filter_relation(self, relation):
        if self.freebase_prefix not in relation:
            return False
        a = relation.replace(self.freebase_prefix, "")
        if a.startswith("m.") or a.startswith("en.") or a.startswith("type.") or a.startswith("common.") or a.startswith("freebase."):
            return False
        if a not in kb_relations:
            return False
        return a

    def filter_entity(self, entity):
        if entity:
            if self.freebase_prefix not in entity:
                return False
            a = entity.replace(self.freebase_prefix, "")
            if a.startswith("m.") or a.startswith("en.") or a.startswith("g."):
                return a
            else:
                return False
        else:
            return False

    def get_p_o(self, s):
        '''获取s, 出边信息
        :return p_o_set, o_set, p_set
        '''
        p_o_set = set()
        o_set = set()
        p_set = set()
        results = self.freebase_sparql.execute(self.prefix + """ SELECT DISTINCT ?p ?o  WHERE { 
                       """ + self.return_str_not_something("?p") + """{:""" + s + """ ?p ?o . }}""")
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

    def get_p_o_literal(self, s):
        '''获取s, 出边信息
        :return p_o_set, o_set, p_set
        '''
        p_o_set = set()
        o_set = set()
        p_set = set()
        results = self.freebase_sparql.execute(self.prefix + """ SELECT DISTINCT ?p ?o  WHERE { 
                       """ + self.return_str_not_something("?p") + """{:""" + s + """ ?p ?o . }}""")
        for result in results:
            p = result[0]
            o = result[1]
            p = self.filter_relation(p)
            if p:
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
        results = self.freebase_sparql.execute(self.prefix + """SELECT DISTINCT ?s ?p  WHERE {
         """ + self.return_str_not_something("?p") + """{?s  ?p :""" + o + """ . }}""")
        for result in results:
            s = result[0]
            p = result[1]
            s = self.filter_entity(s)
            p = self.filter_relation(p)
            if s and p:
                s_p_set.add("\t".join([s, p]))
                s_set.add(s)
                p_set.add(p)
        return s_p_set, s_set, p_set

    def get_s_p_literal_none(self, literal):
        '''
        sqlodbc.get_s_p_literal_none('"1805"^^xsd:dateTime')
        读取literal的入边信息
        :return s_p_set
        '''
        s_p_set = set()
        s_set = set()
        p_set = set()
        results = self.freebase_sparql.execute(self.prefix + """ SELECT DISTINCT ?s ?p  WHERE {
         VALUES ?x1 { """ + literal + """ } . ?s ?p ?x1 . } """)
        for result in results:
            s = result[0]
            p = result[1]
            s = self.filter_entity(s)
            p = self.filter_relation(p)
            if s and p:
                s_p_set.add("\t".join([s, p]))
                s_set.add(s)
                p_set.add(p)
        return s_p_set, s_set, p_set

    def get_s_p_literal_function(self, literal, function, literaltype):
        ''':return s_p_set'''
        if function is None:
            function = ''
        s_p_set = set()
        if literaltype is None:
            results = self.freebase_sparql.execute(self.prefix + """ SELECT DISTINCT ?s ?p  WHERE { FILTER (?x1 """ + function + literal + """ ) .
                                                    ?s ?p ?x1 .} limit 100000""")
        else:
            results = self.freebase_sparql.execute(self.prefix + """ SELECT DISTINCT ?s ?p  WHERE {  FILTER (?x1 """ + function + literal + """ ) . 
                                                    ?p :type.property.expected_type :""" + literaltype + """ .
                                                     ?s ?p ?x1 .} limit 100000""")
        i = 0
        j = 0
        for result in results:
            if i < 10000:
                i += 1
            else:
                j += 1
                i = 0
                print(j)
                print(time.strftime('%Y.%m.%d %H:%M:%S ', time.localtime(time.time())))
            s = result[0]
            p = result[1]
            s = self.filter_entity(s)
            p = self.filter_relation(p)
            if s and p:
                s_p_set.add("\t".join([s, p]))
        return s_p_set

    def execute_sparql(self, sparqlquery):
        results = self.freebase_sparql.execute(self.prefix+sparqlquery)
        answers = set()
        for result in results:
            if isinstance(result[0], str) and self.freebase_prefix in result[0]:
                answers.add(result[0].replace(self.freebase_prefix, ""))
            else:
                answers.add(result[0])
        return answers

    def execute_sparql_two_args(self, sparqlquery):
        '''return two args'''
        results = self.freebase_sparql.execute(self.prefix+sparqlquery)
        answers = set()
        for result in results:
            instance = result[0]
            if isinstance(instance, str) and self.freebase_prefix in instance:
                instance = instance.replace(self.freebase_prefix, "")
            class_str = result[1]
            if isinstance(class_str, str) and self.freebase_prefix in class_str:
                class_str = class_str.replace(self.freebase_prefix, "")
            answers.add(('%s\t%s')%(instance, class_str))
        return answers

    def execute_sparql_three_args(self, sparqlquery):
        '''return three args'''
        results = self.freebase_sparql.execute(self.prefix+sparqlquery)
        for result in results:
            instance = result[0]
            if isinstance(instance, str) and self.freebase_prefix in instance:
                instance = instance.replace(self.freebase_prefix, "")
            p_str = result[1]
            if isinstance(p_str, str) and self.freebase_prefix in p_str:
                p_str = p_str.replace(self.freebase_prefix, "")
            o_str = result[2]
            if isinstance(o_str, str) and self.freebase_prefix in o_str:
                o_str = o_str.replace(self.freebase_prefix, "")
            print(('%s\t%s\t%s')%(instance, p_str, o_str))

    def get_p1_p2_p3_by_entity(self, s, label):
        '''+++, +--, .....'''
        p1_p2_p3_set = set()
        sparql_query = """ SELECT DISTINCT  ?p1,?p2,?p3  WHERE { """ \
                       + self.return_str_not_something("?p1") + self.return_str_not_something("?p2") + self.return_str_not_something("?p3")
        if label[0] == '+':
            sparql_query += """ { :""" + s + """ ?p1  ?o1 . """
        else:
            sparql_query += """ { ?o1 ?p1  :""" + s + """  . """

        if label[1] == '+':
            sparql_query += """ ?o1 ?p2 ?o2 . """
        else:
            sparql_query += """ ?o2 ?p2 ?o1 . """

        if label[2] == '+':
            sparql_query += """ ?o2 ?p3 ?o3 . }}"""
        else:
            sparql_query += """ ?o3 ?p3 ?o2 . }}"""

        results = self.freebase_sparql.execute(self.prefix + sparql_query)
        for result in results:
            p1 = result[0]
            p2 = result[1]
            p3 = result[2]
            p1 = self.filter_relation(p1)
            p2 = self.filter_relation(p2)
            p3 = self.filter_relation(p3)
            if p1 and p2 and p3:
                p1_p2_p3_set.add("\t".join([p1, p2, p3]))
        return p1_p2_p3_set

    def get_p1_p2_p3_by_e1_e2(self, e1, e2, label, left_or_right_insertNode='left'):
        '''
        +++, +--, ...
        :param e1:
        :param e2:
        :param label:
        :return:
        '''
        p1_p2_p3_set = set()
        sparql_query = """ SELECT DISTINCT  ?p1,?p2,?p3  WHERE { """ \
                       + self.return_str_not_something("?p1") + self.return_str_not_something("?p2") + self.return_str_not_something("?p3")
        if label[0] == '+':
            sparql_query += """ { :""" + e1 + """ ?p1  ?o1 . """
        else:
            sparql_query += """ { ?o1 ?p1  :""" + e1 + """  . """

        if left_or_right_insertNode == 'left':
            if label[1] == '+':
                sparql_query += """  ?o1 ?p2  ?o2 . """
            else: #-
                sparql_query += """  ?o2 ?p2  ?o1  . """
        else:
            if label[1] == '+':
                sparql_query += """  ?o2 ?p2  ?o1  . """
            else:
                sparql_query += """  ?o1 ?p2  ?o2 . """

        if label[2] == '+':
            sparql_query += """ :""" + e2 + """ ?p3 ?o2 . }}"""
        else:
            sparql_query += """ ?o2 ?p3 :""" + e2 + """ . }}"""
        results = self.freebase_sparql.execute(self.prefix + sparql_query)
        for result in results:
            p1 = result[0]
            p2 = result[1]
            p3 = result[2]
            p1 = self.filter_relation(p1)
            p2 = self.filter_relation(p2)
            p3 = self.filter_relation(p3)
            if p1 and p2 and p3:
                p1_p2_p3_set.add("\t".join([p1, p2, p3]))
        return p1_p2_p3_set

    def get_p1_p2_and_p3_p4_by_e1_e2(self, e1, e2, label):
        '''
         :param e1:
         :param e2:
         :param label:
         :return: p1_p2_set
         '''
        p1_p2_and_p3_p4_set = set()
        sparql_query = """ SELECT DISTINCT ?p1,?p2,?p3,?p4 WHERE { """ \
                       + self.return_str_not_something("?p1") + self.return_str_not_something("?p2") \
                       + self.return_str_not_something("?p3") + self.return_str_not_something("?p4")
        if label[0] == '+':
            sparql_query += """ { :""" + e1 + """ ?p1  ?o1 . """
        else:
            sparql_query += """ { ?o1 ?p1  :""" + e1 + """  . """
        if label[1] == '+':
            sparql_query += """ ?o1 ?p2 ?o2 . """
        else:
            sparql_query += """ ?o2 ?p2 ?o1 . """

        if label[2] == '+':
            sparql_query += """ ?o3 ?p3 ?o2 ."""
        else:
            sparql_query += """ ?o2 ?p3 ?o3 ."""

        if label[3] == '+':
            sparql_query += """ :""" + e1 + """ ?p4 ?o3 . }}"""
        else:
            sparql_query += """ ?o3 ?p4 :""" + e1 + """ . }}"""

        results = self.freebase_sparql.execute(self.prefix + sparql_query)
        for result in results:
            p1 = result[0]
            p2 = result[1]
            p3 = result[2]
            p4 = result[3]
            p1 = self.filter_relation(p1)
            p2 = self.filter_relation(p2)
            p3 = self.filter_relation(p3)
            p4 = self.filter_relation(p4)
            if p1 and p2 and p3 and p4:
                p1_p2_and_p3_p4_set.add("\t".join([p1, p2, p3, p4]))
        return p1_p2_and_p3_p4_set

    def get_p1_p2_p3_p4_by_e1(self, s, label):
        '''+++, +--, .....'''
        p1_p2_p3_p4_set = set()
        sparql_query = """ SELECT DISTINCT  ?p1,?p2,?p3,?p4  WHERE { """ \
                       + self.return_str_not_something("?p1") + self.return_str_not_something("?p2") \
                       + self.return_str_not_something("?p3") + self.return_str_not_something("?p4")
        if label[0] == '+':
            sparql_query += """ { :""" + s + """ ?p1  ?o1 . """
        else:
            sparql_query += """ { ?o1 ?p1  :""" + s + """  . """

        if label[1] == '+':
            sparql_query += """ ?o1 ?p2 ?o2 . """
        else:
            sparql_query += """ ?o2 ?p2 ?o1 . """

        if label[2] == '+':
            sparql_query += """ ?o2 ?p3 ?o3 . """
        else:
            sparql_query += """ ?o3 ?p3 ?o2 . """

        if label[3] == '+':
            sparql_query += """ ?o3 ?p4 ?o4 . }} """
        else:
            sparql_query += """ ?o4 ?p4 ?o3 . }} """
        results = self.freebase_sparql.execute(self.prefix + sparql_query)
        for result in results:
            p1 = result[0]
            p2 = result[1]
            p3 = result[2]
            p4 = result[3]
            p1 = self.filter_relation(p1)
            p2 = self.filter_relation(p2)
            p3 = self.filter_relation(p3)
            p4 = self.filter_relation(p4)
            if p1 and p2 and p3 and p4:
                p1_p2_p3_p4_set.add("\t".join([p1, p2, p3, p4]))
        return p1_p2_p3_p4_set


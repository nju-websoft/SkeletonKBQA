from SPARQLWrapper import SPARQLWrapper2, JSON, SPARQLWrapper, GET

from common.globals_args import argument_parser


class SparqlQueryHTML():

    def __init__(self):
        self.freebase_sparql = SPARQLWrapper(argument_parser.freebase_sparql_html)
        self.freebase_prefix = "http://rdf.freebase.com/ns/"
        self.prefix = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>" \
                      "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>" \
                      "PREFIX : <http://rdf.freebase.com/ns/>"

    def filter_entity(self,entity):
        if self.freebase_prefix not in entity:
            return False
        a = entity.replace(self.freebase_prefix,"")
        if a.startswith("m.") or a.startswith("en.") :
            return a
        else:
            return False

    def execute_sparql(self,sparqlquery):
        self.freebase_sparql.setQuery(self.prefix+sparqlquery)
        self.freebase_sparql.setReturnFormat(JSON)
        try:
            results = self.freebase_sparql.query().convert()
        except Exception:
            return set()
        answers=set()
        if len(results["results"]["bindings"])==1:
            result=results["results"]["bindings"][0]
            if result["value"]["type"]=="typed-literal":
                answers.add(result["value"]["value"])
                return answers
        for result in results["results"]["bindings"]:
            answers.add(self.filter_entity(result["value"]["value"]))

        return answers

    def get_name(self,entity):
        sparqlquery = """SELECT (?name AS ?value) WHERE {
        SELECT DISTINCT ?name WHERE {
        VALUES ?x0 { """+entity+""" } .
        ?x0:type.object.name ?name .
        }
        }
        """
        sparqlquery = """SELECT (?name AS ?value) WHERE {
                SELECT DISTINCT ?name  WHERE {
                VALUES ?x0 { :""" + entity + """ } .
                ?x0:type.object.name ?name .
                FILTER (langMatches(lang(?name), 'en')).
                }}"""
        answers = set()
        self.freebase_sparql.setQuery(self.prefix + sparqlquery)
        self.freebase_sparql.setReturnFormat(JSON)
        results = self.freebase_sparql.query().convert()
        # print (results) #{'head': {'link': [], 'vars': ['value']}, 'results': {'distinct': False, 'ordered': True, 'bindings': [{'value': {'type': 'literal', 'xml:lang': 'en', 'value': 'Abraham Lincoln'}}]}}
        for result in results["results"]["bindings"]:
            answers.add(self.filter_entity(result["value"]["value"]))
        return answers

    def get_names_dict(self,entity_set):
        result_names_dict = dict()
        for entity in entity_set:
            sparqlquery="""SELECT (?name AS ?value) WHERE {
                    SELECT DISTINCT ?name  WHERE {
                    VALUES ?x0 { :""" + entity + """ } .
                    ?x0:type.object.name ?name .
                    FILTER (langMatches(lang(?name), 'en')).
                    }}"""
            self.freebase_sparql.setQuery(self.prefix + sparqlquery)
            self.freebase_sparql.setReturnFormat(JSON)
            results = self.freebase_sparql.query().convert()
            # print (results) #{'head': {'link': [], 'vars': ['value']},
            # 'results': {'distinct': False, 'ordered': True, 'bindings': [{'value': {'type': 'literal', 'xml:lang': 'en', 'value': 'Abraham Lincoln'}}]}}
            for result in results["results"]["bindings"]:
                result_names_dict[entity] = result["value"]["value"]
        return result_names_dict

    def get_names(self,entity_set):
        result_names = set()
        for entity in entity_set:
            sparqlquery="""SELECT (?name AS ?value) WHERE {
                    SELECT DISTINCT ?name  WHERE {
                    VALUES ?x0 { :""" + entity + """ } .
                    ?x0:type.object.name ?name .
                    FILTER (langMatches(lang(?name), 'en')).
                    }}"""
            self.freebase_sparql.setQuery(self.prefix + sparqlquery)
            self.freebase_sparql.setReturnFormat(JSON)
            results = self.freebase_sparql.query().convert()
            # print (results) #{'head': {'link': [], 'vars': ['value']},
            # 'results': {'distinct': False, 'ordered': True, 'bindings': [{'value': {'type': 'literal', 'xml:lang': 'en', 'value': 'Abraham Lincoln'}}]}}
            for result in results["results"]["bindings"]:
                result_names.add(result["value"]["value"])
        return result_names


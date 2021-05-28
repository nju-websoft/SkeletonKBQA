from common_structs.structure import Structure


class QuestionAnnotation():

    def __init__(self,
                 qid=None,
                 question=None,
                 question_normal=None,
                 tokens=None,
                 span_tree=None,
                 span_tree_hybrid_dependency_graph=None,
                 surface_tokens_to_dep_node_dict=None,
                 main_ungrounded_graph=None,
                 sequence_ner_tag_dict=None,
                 gold_graph_query=None,
                 gold_answer=None,
                 gold_sparql_query=None,
                 compositionality_type=None,
                 q_function=None):
        self.qid = qid
        self.question = question
        self.question_normal = question_normal
        if span_tree is not None:
            self.tokens = span_tree.tokens
        else:
            self.tokens = tokens
        self.compositionality_type = None
        self.function = None
        self.commonness = 0
        self.span_tree = span_tree
        self.span_tree_hybrid_dependency_graph = span_tree_hybrid_dependency_graph
        self.surface_tokens_to_dep_node_dict = surface_tokens_to_dep_node_dict
        self.sequence_ner_tag_dict = sequence_ner_tag_dict
        self.main_ungrounded_graph = main_ungrounded_graph
        self.num_node = len(self.main_ungrounded_graph.nodes)
        self.num_edge = len(self.main_ungrounded_graph.edges)

        self.abstract_question_word = []
        self.important_words_list = []
        self.gold_graph_query = gold_graph_query
        self.gold_answer = gold_answer
        self.gold_sparql_query = gold_sparql_query

        self.compositionality_type = compositionality_type
        self.function = q_function

    def convert_to_structure(self):
        '''
        structure = Structure(question_annotation.qid,
                      question_annotation.question_normal,
                      words=str([question_annotation.tokens[i].value for i in range(len(question_annotation.tokens))]),
                      function=question_annotation.function,
                      compositionality_type=question_annotation.compositionality_type,
                      num_node=len(super_ungrounded_graph.nodes),
                      num_edge=len(super_ungrounded_graph.edges),
                      span_tree=str(question_annotation.span_tree),
                      gold_graph_query=question_annotation.gold_graph_query,
                      gold_answer=question_annotation.gold_answer,
                      gold_sparql_query=question_annotation.gold_sparql_query)
        '''
        return Structure(self.qid,
                              self.question_normal,
                              words=str([self.tokens[i].value for i in range(len(self.tokens))]),
                              function=self.function,
                              compositionality_type=self.compositionality_type,
                              num_node=self.num_node,
                              num_edge=self.num_edge,
                              span_tree=str(self.span_tree),
                              gold_graph_query=self.gold_graph_query,
                              gold_answer=self.gold_answer,
                              gold_sparql_query=self.gold_sparql_query)


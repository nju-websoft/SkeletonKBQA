from method_sp.parsing import parsing_utils
from skeleton_parsing import fine_grained_dependency
from skeleton_parsing import skeleton_parser_token
from skeleton_parsing.skeleton_args import nltk_nlp


def get_hybrid_dependency_tree(question):
    tokens = parsing_utils.create_tokens(question.split(" "))
    span_tree = skeleton_parser_token.span_tree_generation_head(tokens=tokens)
    print('#question:\t', question)
    print('#span tree:\t', span_tree)
    span_tree_hybrid_dependency_graph = fine_grained_dependency.span_tree_to_hybrid_dependency_graph_interface(span_tree=span_tree)
    span_tree_hybrid_dependency_graph = parsing_utils.update_dependencygraph_indexs(old_dependency_graph=span_tree_hybrid_dependency_graph)
    return span_tree_hybrid_dependency_graph


def get_dependency_tree(question):
    dependency_graph = nltk_nlp.generate_dependency_graph(question)
    return dependency_graph


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-inputquestion', action='store', type=str, help='input your questions', default='Where was the person who spoke about the Berlin Wall raised ?')
    args = parser.parse_args()
    dependency_tree = get_hybrid_dependency_tree(question=args.inputquestion)


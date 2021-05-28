from common.globals_args import q_mode


if q_mode in ['cwq', 'graphq']:
    from datasets_interface.virtuoso_interface import freebase_kb_interface as kb_interface
elif q_mode in ['lcquad']:
    from datasets_interface.virtuoso_interface import dbpedia_kb_interface as kb_interface


def set_denotation(grounded_graph, q_compositionality_type):
    denotation = grounded_graph.denotation
    if q_compositionality_type == 'ask':
        denotation = [1]
    elif q_compositionality_type == 'count':
        num = 0
        if denotation is not None:
            num = len(denotation)
        denotation = [num]
    return denotation


def sparql_to_denotation(sparqlquery):
    denotation = []
    if q_mode in ['cwq', 'graphq']:
        denotation = list(kb_interface.execute_sparql(sparqlquery))
    elif q_mode in ['lcquad']:
        denotation = list(kb_interface.execute_sparql(sparqlquery))
    return denotation

from question_classification import qtype_classifier
from question_classification import qtype_rules
from common.globals_args import q_mode

if q_mode == 'cwq':
    from question_classification import qcomparative_classifier, qsuperlative_classifier


def question_type_interface(question_normal, dataset='lcquad'):
    assert dataset in ['graphq', 'lcquad', 'cwq']
    compositionality_type = None
    q_function = None
    if dataset == 'graphq':
        compositionality_type, q_function = _question_type_graphq(question_normal=question_normal)
    elif dataset == 'lcquad':
        compositionality_type, q_function = _question_type_lcquad(question_normal=question_normal)
    elif dataset == 'cwq':
        compositionality_type, q_function = _question_type_cwq(question_normal=question_normal)
    return compositionality_type, q_function


def _question_type_graphq(question_normal):
    compositionality_type = qtype_classifier.process(question_normal)
    q_function = None
    assert compositionality_type in ["bgp", "count", "superlative", "comparative"]
    if compositionality_type in ['bgp', 'count']:
        pass
    elif compositionality_type in ['superlative']:
        q_function = qtype_rules.get_superlative_type(question_normal)
    elif compositionality_type in ['comparative']:
        q_function = qtype_rules.get_comparative_type(question_normal)
    return compositionality_type, q_function


def _question_type_lcquad(question_normal):
    compositionality_type = qtype_classifier.process(question_normal)
    q_function = None
    return compositionality_type, q_function


def _question_type_cwq(question_normal):
    compositionality_type = qtype_classifier.process(question_normal)
    q_function = None
    assert compositionality_type in ['composition', 'conjunction', 'comparative', 'superlative']
    if compositionality_type in ['composition', 'conjunction']:
        pass
    elif compositionality_type in ['superlative']:
        q_function = qsuperlative_classifier.process(question_normal)
    elif compositionality_type in ['comparative']:
        q_function = qcomparative_classifier.process(question_normal)
    return compositionality_type, q_function


def aggregation_interface(question_normal):
    '''
    no use
    rule-based classifier
    # is_agg, serialization_list = aggregation_process.aggregation_interface(question_normal=question_normal)
    '''
    question_normal = question_normal.lower()
    count_serialization_list = qtype_rules.count_serialization(question=question_normal)
    is_count = qtype_rules.is_count_funct(count_serialization_list)
    superlative_serialization_list = qtype_rules.superlative_serialization(question=question_normal)
    is_superlative = qtype_rules.is_superlative_funct(superlative_serialization_list)
    comparative_serialization_list = qtype_rules.comparative_serialization(question=question_normal)
    is_comparative = qtype_rules.is_comparative_funct(comparative_serialization_list)
    if is_count:
        return 'count', count_serialization_list
    elif is_superlative:
        return 'superlative', superlative_serialization_list
    elif is_comparative:
        return 'comparative', comparative_serialization_list
    else:
        return 'none', None



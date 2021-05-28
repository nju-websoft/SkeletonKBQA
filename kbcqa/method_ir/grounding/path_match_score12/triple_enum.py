import copy


def get_reverse_triples_by_triples(triples, property_reverse_dict):

    def _get_reverse_property_from_lexcion(property, property_reverse_dict):
        '''get reverse property'''
        reverse_property = None
        if property in property_reverse_dict.keys():
            reverse_property = property_reverse_dict[property][0]
        for key, value in property_reverse_dict.items():
            if key == property:
                reverse_property = value[0]
            if value[0] == property:
                reverse_property = key
        return reverse_property

    new_triples = []
    for triple in triples:
        reverse_predicate = _get_reverse_property_from_lexcion(property=triple['predicate'], property_reverse_dict=property_reverse_dict)
        new_triple_dict = dict()
        if reverse_predicate is None:
            new_triples.append(new_triple_dict)
        else:
            new_triple_dict['subject'] = triple['object']
            new_triple_dict['predicate'] = reverse_predicate
            new_triple_dict['object'] = triple['subject']
            new_triples.append(new_triple_dict)
    return new_triples


triple_sequence_completed_list = []
triple_sequence = []


def _recursion_triples(grounding_result_list, index):
    if index == len(grounding_result_list):
        triple_sequence_completed_list.append(triple_sequence.copy())
        return
    for triple in grounding_result_list[index]:
        triple_sequence.append(triple)
        _recursion_triples(grounding_result_list, index+1)
        triple_sequence.pop()


def get_all_reverse_triples(triples, property_reverse_dict):
    reverse_triples_ = get_reverse_triples_by_triples(triples=triples, property_reverse_dict=property_reverse_dict)
    grounding_result_list = []
    for triple, reverse_triple in zip(triples, reverse_triples_):
        if len(reverse_triple) == 0:
            grounding_result_list.append([triple])
        else:
            grounding_result_list.append([triple, reverse_triple])
    triple_sequence_completed_list.clear()
    _recursion_triples(grounding_result_list=grounding_result_list, index=0)
    return copy.deepcopy(triple_sequence_completed_list)


"""
triples = []
triple_dict = {}
triple_dict['subject'] = 'http://rdf.freebase.com/ns/en.christianity'
triple_dict['predicate'] = 'broadcast.genre.content'
triple_dict['object'] = '?x_1'
triples.append(triple_dict)
triple_dict_2 = {}
triple_dict_2['subject'] = '?uri_0'
triple_dict_2['predicate'] = 'broadcast.broadcast.content'
triple_dict_2['object'] = '?x_1'
triples.append(triple_dict_2)
from method_sp.grounding.grounding_args import property_reverse_dict
triples_list = get_all_reverse_triples(triples=triples, property_reverse_dict=property_reverse_dict)
for triples in triples_list:
    print(triples)
"""


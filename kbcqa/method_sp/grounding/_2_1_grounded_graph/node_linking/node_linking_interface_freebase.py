from collections import OrderedDict
from fuzzywuzzy import fuzz
from method_sp.grounding._2_1_grounded_graph.node_linking import node_linking_args
from method_sp.grounding._2_1_grounded_graph.node_linking import node_linking_utils


if node_linking_args.kb_mode == 'kb_freebase_latest':  #aqqu entity linking
    from method_sp.grounding._2_1_grounded_graph.node_linking.entity_linking_aqqu_vocab.surface_index_memory import EntitySurfaceIndexMemory
    elp = EntitySurfaceIndexMemory(entity_list_file=node_linking_args.entity_list_file,
                                   surface_map_file=node_linking_args.surface_map_file,
                                   entity_index_prefix=node_linking_args.entity_index_prefix)
else:   # graphquestion, webq, zhang's lexicon
    from method_sp.grounding._2_1_grounded_graph.node_linking.entity_linking_en_vocab.entity_link_pipeline import EntityLinkPipeline
    elp = EntityLinkPipeline(freebase_graph_name_entity_file=node_linking_args.freebase_graph_name_entity_file,
                         freebase_graph_alias_entity_file=node_linking_args.freebase_graph_alias_entity_file,
                         graphquestions_train_friendlyname_entity_file=node_linking_args.graphquestions_train_friendlyname_entity_file,
                         clueweb_mention_pro_entity_file=node_linking_args.clueweb_mention_pro_entity_file)


def node_linking(qid, node=None, top_k=10):
    '''node grounding
    {'entity_id', 'entity_pro'}
    return top_k results'''
    results_dict = OrderedDict()
    if node.node_type == 'entity':
        entities_pros = elp.get_indexrange_entity_el_pro_one_mention(node_linking_utils.get_old_mention(node.friendly_name).lower(), top_k=top_k)
        if len(entities_pros) == 0:
            entities_pros = _node_entity_linking_quotation(node_mention=node.friendly_name, top_k=1)
        for entity, pro in entities_pros.items():
            results_dict[entity] = pro
    elif node.node_type == 'class':
        results_dict['hhh'] = 1.0  # results_dict = cl_interface(node=node) 2020.11.03
    elif node.node_type == 'literal' or node.node_type == 'DATE':
        if node.normalization_value is not None:
            results_dict[node.normalization_value] = 1.0
        else:
            results_dict[node.friendly_name] = 1.0
        node.node_type = 'literal'
    return results_dict


def _node_entity_linking_quotation(node_mention, top_k, threshold=0.6):
    '''quotation, 利用求最大公共子序列来linking'''
    entities_pros_dict = dict()
    for entity, quotation_value in node_linking_args.quotation_dict.items():
        if len(quotation_value) == 0: continue
        if node_mention == quotation_value[0]:
            entities_pros_dict[entity] = 1.0
        else:
            score = fuzz.ratio(node_mention, quotation_value[0]) / 100
            if score > threshold:
                entities_pros_dict[entity] = score
    results_list = sorted(entities_pros_dict.items(), key=lambda d: d[1], reverse=True)
    if len(results_list) > top_k:
        results_list = results_list[0:top_k]
    return dict(results_list)

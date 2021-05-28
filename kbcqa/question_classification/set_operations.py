from common.hand_files import write_structure_file, read_structure_file
import os
import time
from question_classification import operation_utils


def run_grounding_graph_update_denotation_graphq(input_file_folder):
    for path in os.listdir(input_file_folder):
        structure_with_grounded_graphq_file = input_file_folder + path
        print(path)
        structure_list = read_structure_file(structure_with_grounded_graphq_file)
        is_aggregation = False
        for structure in structure_list:
            qtype = structure.compositionality_type
            assert qtype in ["bgp","count","superlative","comparative"]
            if qtype in ['bgp', 'count']:
                continue
            is_aggregation = True
            for j, ungrounded_graph in enumerate(structure.ungrounded_graph_forest):
                if j != len(structure.ungrounded_graph_forest) - 1:
                    continue
                grounded_graph_list = ungrounded_graph.get_grounded_graph_forest()

                if qtype in ['superlative']:
                    superlative_q_type = structure.function
                    for grounded_graph in grounded_graph_list:
                        print(grounded_graph.grounded_query_id)
                        sample_value = grounded_graph.denotation[0][1]
                        denotation_length = len(grounded_graph.denotation)
                        new_denotation = set()
                        try:
                            if denotation_length > 1 and operation_utils.is_number(sample_value):
                                start = time.clock()
                                new_denotation.add(operation_utils.get_denotation_with_superlativeconstraint_float(
                                                                denotation_with_value_list=grounded_graph.denotation, superlative_type=superlative_q_type))
                                end = time.clock()
                                print('get_denotation_with_superlativeconstraint_float:\t', end - start)
                            elif denotation_length > 1 and operation_utils.isVaildDate(sample_value):
                                start = time.clock()
                                new_denotation.add(operation_utils.get_denotation_with_superlativeconstraint_datetime(
                                                                denotation_with_value_list=grounded_graph.denotation, superlative_type=superlative_q_type))
                                end = time.clock()
                                print('get_denotation_with_superlativeconstraint_datetime:\t', end - start)
                            else:
                                start = time.clock()
                                for answer in grounded_graph.denotation:
                                    if type(answer) == list:
                                        new_denotation.add(answer[0])
                                    else:
                                        new_denotation.add(answer)
                                end = time.clock()
                                print('grounded_graph.denotation1:\t', end - start)
                        except Exception as e:
                            start = time.clock()
                            for answer in grounded_graph.denotation:
                                if type(answer) == list:
                                    new_denotation.add(answer[0])
                                else:
                                    new_denotation.add(answer)
                            end = time.clock()
                            print('grounded_graph.denotation2:\t', end - start)
                        grounded_graph.denotation = list(new_denotation)

                elif qtype in ['comparative']:
                    q_function = structure.function
                    normalization_value = None
                    for ungrounded_node in ungrounded_graph.nodes:
                        if ungrounded_node.normalization_value is not None:
                            normalization_value = ungrounded_node.normalization_value
                            break
                    for grounded_graph in grounded_graph_list:
                        denotation_length = len(grounded_graph.denotation)
                        try:
                            if denotation_length > 1 and ('^^xsd:dateTime' in normalization_value or '^^http://www.w3.org/2001/XMLSchema#datetime' in normalization_value):
                                new_denotation = operation_utils.filter_by_datetime_compare(denotation=grounded_graph.denotation,
                                                                            compare_element=normalization_value.split('^^')[0], q_function=q_function)
                            elif denotation_length > 1:
                                # composition[['1768.0^^http://www.w3.org/2001/XMLSchema#double', 'literal']]
                                # composition[['1^^http://www.w3.org/2001/XMLSchema#int', 'literal']]
                                if '^^' in normalization_value:
                                    normalization_value = normalization_value.split('^^')[0]
                                new_denotation = operation_utils.filter_by_float_compare(denotation=grounded_graph.denotation,
                                                                         compare_element=normalization_value, q_function=q_function)
                            else:
                                all_denotation = set()
                                for answer in grounded_graph.denotation:
                                    if type(answer) == list:
                                        all_denotation.add(answer[0])
                                    else:
                                        all_denotation.add(answer)
                                new_denotation = list(all_denotation)
                            grounded_graph.denotation = new_denotation
                        except Exception as e:
                            all_denotation = set()
                            for answer in grounded_graph.denotation:
                                if type(answer) == list:
                                    all_denotation.add(answer[0])
                                else:
                                    all_denotation.add(answer)
                            print('error', grounded_graph.grounded_query_id, grounded_graph.denotation)
                            grounded_graph.denotation = list(all_denotation)

        if is_aggregation:
            write_structure_file(structure_list, structure_with_grounded_graphq_file)
    print('over')


def run_grounding_graph_update_denotation_cwq(input_file_folder):
    for path in os.listdir(input_file_folder):
        structure_with_grounded_graphq_file = input_file_folder + path
        print(path)
        structure_list = read_structure_file(structure_with_grounded_graphq_file)
        is_aggregation = False
        for structure in structure_list:
            qtype = structure.compositionality_type
            assert qtype in ['composition', 'conjunction', 'comparative', 'superlative']
            if qtype in ['composition', 'conjunction']:
                continue
            for j, ungrounded_graph in enumerate(structure.ungrounded_graph_forest):
                if j != len(structure.ungrounded_graph_forest) - 1:
                    continue
                grounded_graph_list = ungrounded_graph.get_grounded_graph_forest()
                if qtype in ['superlative']:
                    is_aggregation = True
                    superlative_q_type = structure.function
                    for grounded_graph in grounded_graph_list:
                        print(grounded_graph.grounded_query_id)
                        sample_value = grounded_graph.denotation[0][1]
                        denotation_length = len(grounded_graph.denotation)
                        new_denotation = set()
                        try:
                            if denotation_length > 1 and operation_utils.is_number(sample_value):
                                start = time.clock()
                                new_denotation.add(operation_utils.get_denotation_with_superlativeconstraint_float(
                                                    denotation_with_value_list=grounded_graph.denotation, superlative_type=superlative_q_type))
                                end = time.clock()
                                print('get_denotation_with_superlativeconstraint_float:\t', end - start)
                            elif denotation_length > 1 and operation_utils.isVaildDate(sample_value):
                                start = time.clock()
                                new_denotation.add(operation_utils.get_denotation_with_superlativeconstraint_datetime(
                                                    denotation_with_value_list=grounded_graph.denotation, superlative_type=superlative_q_type))
                                end = time.clock()
                                print('get_denotation_with_superlativeconstraint_datetime:\t', end - start)
                            else:
                                start = time.clock()
                                for answer in grounded_graph.denotation:
                                    if type(answer) == list:
                                        new_denotation.add(answer[0])
                                    else:
                                        new_denotation.add(answer)
                                end = time.clock()
                                print('grounded_graph.denotation1:\t', end - start)
                        except Exception as e:
                            start = time.clock()
                            for answer in grounded_graph.denotation:
                                if type(answer) == list:
                                    new_denotation.add(answer[0])
                                else:
                                    new_denotation.add(answer)
                            end = time.clock()
                            print('grounded_graph.denotation2:\t', end - start)
                        grounded_graph.denotation = list(new_denotation)

                elif qtype in ['comparative']:
                    is_aggregation = True
                    q_function = structure.function
                    normalization_value = None
                    for ungrounded_node in ungrounded_graph.nodes:
                        if ungrounded_node.normalization_value is not None:
                            normalization_value = ungrounded_node.normalization_value
                            break
                    for grounded_graph in grounded_graph_list:
                        denotation_length = len(grounded_graph.denotation)
                        try:
                            if denotation_length > 1 and ('^^xsd:dateTime' in normalization_value or '^^http://www.w3.org/2001/XMLSchema#datetime' in normalization_value):
                                new_denotation = operation_utils.filter_by_datetime_compare(denotation=grounded_graph.denotation,
                                                                        compare_element=normalization_value.split('^^')[0], q_function=q_function)
                            elif denotation_length > 1:
                                new_denotation = operation_utils.filter_by_float_compare(denotation=grounded_graph.denotation,
                                                                        compare_element=normalization_value, q_function=q_function)
                            else:
                                all_denotation = set()
                                for answer in grounded_graph.denotation:
                                    if type(answer) == list:
                                        all_denotation.add(answer[0])
                                    else:
                                        all_denotation.add(answer)
                                new_denotation = list(all_denotation)
                            grounded_graph.denotation = new_denotation
                        except Exception as e:
                            all_denotation = set()
                            for answer in grounded_graph.denotation:
                                if type(answer) == list:
                                    all_denotation.add(answer[0])
                                else:
                                    all_denotation.add(answer)
                            print('error', grounded_graph.grounded_query_id, grounded_graph.denotation)
                            grounded_graph.denotation = list(all_denotation)

        if is_aggregation:
            write_structure_file(structure_list, structure_with_grounded_graphq_file)

    print('over')


from datetime import datetime
import re


def filter_by_float_compare(denotation, compare_element, q_function):
    result_denotation = set()
    for answer, literal in denotation:
        if q_function == '>':
            if float(literal) > float(compare_element):
                result_denotation.add(answer)
        elif q_function == '<':
            if float(literal) < float(compare_element):
                result_denotation.add(answer)
        else:
            result_denotation.add(answer)
    return list(result_denotation)


def convert_to_time(element):
    if ":" in element:
        element_datetime = datetime.strptime(element, "%Y-%m-%d %H:%M:%S")
    elif '-' in element:
        cols = element.split('-')
        if len(cols) == 2:
            element_datetime = datetime.strptime(element, '%Y-%m')
        elif len(cols) == 3:
            element_datetime = datetime.strptime(element, '%Y-%m-%d')
        else:
            element_datetime = datetime.strptime(cols[0] + '-' + cols[1] + '-' + cols[2], '%Y-%m-%d')
    else:
        element_datetime = datetime.strptime(element, '%Y')
    return element_datetime


def filter_by_datetime_compare(denotation, compare_element, q_function):
    result_denotation = set()
    compare_element_datetime = convert_to_time(compare_element)
    for answer, literal in denotation:
        literal_datetime = convert_to_time(literal)
        if q_function == '>':
            if literal_datetime > compare_element_datetime:
                result_denotation.add(answer)
        elif q_function == '<':
            if literal_datetime < compare_element_datetime:
                result_denotation.add(answer)
        else:
            result_denotation.add(answer)
    return list(result_denotation)


def isVaildDate(date):
    try:
        if ":" in date:
            datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        elif '-' in date:
            cols = date.split('-')
            if len(cols) == 2:
                datetime.strptime(date, '%Y-%m')
            elif len(cols) == 3:
                datetime.strptime(date, '%Y-%m-%d')
            else:
                datetime.strptime(cols[0] + '-' + cols[1] + '-' + cols[2], '%Y-%m-%d')
        else:
            datetime.strptime(date, "%Y")
        return True
    except:
        return False


def is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False


def get_denotation_with_superlativeconstraint_float(denotation_with_value_list, superlative_type):
    values_list = []
    value_to_dentotion_dict = dict()
    for denotation_with_value in denotation_with_value_list:
        float_value = float(denotation_with_value[1])
        values_list.append(float_value)
        value_to_dentotion_dict[float_value] = denotation_with_value[0]
    assert superlative_type in ['argmax', 'argmin']
    if superlative_type == 'argmax':
        values_list.sort(reverse=True)
    elif superlative_type == 'argmin':
        values_list.sort()
    value = values_list[0]
    return value_to_dentotion_dict[value]


def get_denotation_with_superlativeconstraint_datetime(denotation_with_value_list, superlative_type):
    values_list = []
    value_to_dentotion_dict = dict()
    for denotation_with_value in denotation_with_value_list:
        time_value = convert_to_time(denotation_with_value[1])
        values_list.append(time_value)
        value_to_dentotion_dict[time_value] = denotation_with_value[0]
    assert superlative_type in ['argmax', 'argmin']
    if superlative_type == 'argmax':
        values_list.sort(reverse=True)
    elif superlative_type == 'argmin':
        values_list.sort()
    value = values_list[0]
    return value_to_dentotion_dict[value]

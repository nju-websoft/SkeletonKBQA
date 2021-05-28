from datasets_interface.virtuoso_interface import freebase_kb_interface
from common.hand_files import read_json, write_json
from common.globals_args import fn_cwq_file


mid_to_names_dict = read_json(fn_cwq_file.cache_mid_to_names)


def get_names(instance_str):
    if instance_str in mid_to_names_dict:
        mid_dict = mid_to_names_dict[instance_str]
    else:
        mid_dict = dict()
        mid_dict['answer_id'] = instance_str
        if isinstance(instance_str, str): # mid = 'm.02hwgbx'
            labels = freebase_kb_interface.get_names(instance_str)
            mid_dict['answer'] = list(labels)
            alias = freebase_kb_interface.get_alias(instance_str)
            mid_dict['aliases'] = list(alias)
        else:
            mid_dict['answer'] = [instance_str]
            mid_dict['aliases'] = [instance_str]
        mid_to_names_dict[instance_str] = mid_dict
    return mid_dict


def write_cache_json():
    write_json(mid_to_names_dict, fn_cwq_file.cache_mid_to_names)


if __name__ == '__main__':
    cwq_prediction_test_json = read_json('./2020.01.11_output_cwq_IR5_all.json')
    for cwq_test_json in cwq_prediction_test_json:
        answers = []
        for answer_id in cwq_test_json['answers_id']:
            print(cwq_test_json['ID'], '\t', answer_id)
            answers.append(get_names(answer_id))
        cwq_test_json['answers'] = answers

    write_json(cwq_prediction_test_json, './2020.01.11_output_cwq_IR5_all_with_names.json')
    write_json(mid_to_names_dict, './cache_mid_to_names.json')



""" Official evaluation script for v1.0 of the ComplexWebQuestions dataset. """
import unicodedata
import re
import json
import pandas as pd


def proprocess(answer):
    proc_answer = unicodedata.normalize('NFKD', answer).encode('ascii', 'ignore').decode(encoding='UTF-8')
    # removing common endings such as "f.c."
    proc_answer = re.sub(r'\W', ' ', proc_answer).lower().strip()
    # removing The, a, an from begining of answer as proposed by SQuAD dataset answer comparison
    if proc_answer.startswith('the '):
        proc_answer = proc_answer[4:]
    if proc_answer.startswith('a '):
        proc_answer = proc_answer[2:]
    if proc_answer.startswith('an '):
        proc_answer = proc_answer[3:]
    return proc_answer


def compute_P1(matched_answers, golden_answer_list, pred_answer):
    P1 = 0
    if len(matched_answers) > 0:
        P1 = 100
    return P1


def compare_span_to_answer(spans, gold_answers, question, question_annotated=None):
    """ Compares one answers to spans, multiple matches are possible
    spans是预测里面的list
    """
    if len(spans) == 0:
        return []
    found_answers = pd.DataFrame(columns=['span', 'answer', 'span_index'])
    spans_series = pd.Series(spans)
    pre_proc_answers = []
    answers = [answer.lower().strip() for answer in gold_answers]
    for answer in answers:   #遍历标准答案, 处理字符
        pre_proc_answers.append(proprocess(answer))

    question = question.lower().strip()
    # processing question:
    # question_annotated = pd.DataFrame(question_annotated)

    # exact match:
    # 枚举所有的可能,
    for pre_proc_answer, answer in zip(pre_proc_answers, answers):
        if answer in spans: #正确答案在预测里面的list
            exact_match_ind = spans.index(answer)  #确定在预测list的索引
            found_answers = found_answers.append({'span_index': exact_match_ind, 'answer': answer, 'span': answer}, ignore_index=True)

        if pre_proc_answer in spans: #正确答案在预测里面的list
            exact_match_ind = spans.index(pre_proc_answer)
            found_answers = found_answers.append({'span_index': exact_match_ind, 'answer': answer, 'span': pre_proc_answer}, ignore_index=True)

        # year should match year.
        if question.find('year') > -1:
            year_in_answer = re.search('([1-2][0-9]{3})', answer)
            if year_in_answer is not None:
                year_in_answer = year_in_answer.group(0)
            year_spans = spans_series[spans_series == year_in_answer]
            if len(year_spans) > 0:
                found_answers = found_answers.append({'span_index': year_spans.index[0], 'answer': answer, 'span': year_in_answer}, ignore_index=True)

    print('#pre_proc_answers:\t', pre_proc_answers)
    print('#answers:\t', answers)
    print('#spans:\t', spans)
    print('#found_answers:\t', found_answers)
    print('#answers size:\t', len(found_answers))
    print()
    return found_answers.drop_duplicates()


def evaluate(dataset_df, predictions):
    if len(dataset_df) != len(predictions):
        print('predictions file does not match dataset file number of examples!!!')
    P1 = 0
    for index, prediction in enumerate(predictions):  #遍历prediction
        print('#index:\t', index, prediction['ID'])

        golden_answer_list = []  #把ID对应的所有的标准答案的label和aliases都标注出来
        for answer in dataset_df.loc[prediction['ID'], 'answers']:
            golden_answer_list.append(answer['answer'])
            golden_answer_list += answer['aliases']

        question_normal = dataset_df.loc[prediction['ID'], 'question_normal']
        print('#question normal:\t', question_normal)

        curr_P1 = 0
        if not None in golden_answer_list: #如果存在正确的答案, 则统计
            system_answer_names = [prediction['answer']]
            matched_answers = compare_span_to_answer(system_answer_names,
                                                     golden_answer_list,
                                                     dataset_df.loc[prediction['ID'],'question'])

            curr_P1 = compute_P1(matched_answers, golden_answer_list, prediction['answer']) #如果matched_answers > 0, 则P1=100
            P1 += curr_P1
        print('#curr_P1', curr_P1)
        print()
        print()

    print(P1, len(dataset_df))
    return P1/len(dataset_df)


if __name__ == '__main__':
    dataset_file_path = './ComplexWebQuestions_test.json'
    prediction_file_path = './sparqa_results/2021.03.04_output_cwq_IR_9_v0.1_wo_agg_withnames_all_nonull_seed9.json'
    with open(dataset_file_path) as dataset_file:
        dataset_df = pd.DataFrame(json.load(dataset_file)).set_index('ID')
    with open(prediction_file_path) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset_df, predictions)))



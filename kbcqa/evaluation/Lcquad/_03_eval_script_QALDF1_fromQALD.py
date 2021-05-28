from common.hand_files import read_list


class EvaluationCounts():
    '''
    https://github.com/dice-group/gerbil/blob/4d08ad0438688ef11b1bb68e515c344ba9e235d1/src/main/java/org/aksw/gerbil/matching/EvaluationCounts.java
    '''
    def __init__(self, truePositives, falsePositives, falseNegatives):
        self.truePositives = truePositives  #真正例的数量
        self.falsePositives = falsePositives  #假正例的数量
        self.falseNegatives = falseNegatives  #假反例的数量


def calculate_f1_qald(counts):
    '''
    模仿: https://github.com/dice-group/gerbil/blob/4d08ad0438688ef11b1bb68e515c344ba9e235d1/src/main/java/org/aksw/gerbil/evaluate/impl/FMeasureCalculator.java#L158
    计算QALD中对应的F1
    '''
    avgs = []
    avgs.append(0)
    avgs.append(0)
    avgs.append(0)
    for count in counts:
        measures = calculate_measures_qald(count)
        avgs[0] += measures[0]
        avgs[1] += measures[1]
        avgs[2] += measures[2]
    avgs[0] = avgs[0] / len(counts)
    avgs[1] = avgs[1] / len(counts)
    avgs[2] = avgs[2] / len(counts)

    qald_f1 = (2 * avgs[0] * avgs[1]) / (avgs[0] + avgs[1])
    macro_precision = avgs[0]
    macro_recal = avgs[1]
    macro_f1 = avgs[2]
    return macro_precision, macro_recal, macro_f1, qald_f1


def calculate_measures_qald(count):
    '''
    模仿: https://github.com/dice-group/gerbil/blob/4d08ad0438688ef11b1bb68e515c344ba9e235d1/src/main/java/org/aksw/gerbil/evaluate/impl/FMeasureCalculator.java#L158
    '''
    if count.truePositives == 0:
        if count.falsePositives == 0 and count.falseNegatives == 0:
            precision = 1
            recall = 1
            F1_score = 1
        elif count.falsePositives == 0:
            precision = 1
            recall = 0
            F1_score = 0
        else:
            precision = 0
            recall = 0
            F1_score = 0
    else:
        precision = count.truePositives / (count.truePositives + count.falsePositives)
        recall = count.truePositives / (count.truePositives + count.falseNegatives)
        F1_score = (2 * precision * recall) / (precision + recall)
    return precision, recall, F1_score


def calculate(goldList, predictedList):
    truePositives = 0
    falsePositives = 0
    falseNegatives = 0
    for gold in goldList:
        if gold in predictedList:
            truePositives += 1
        else:
            falseNegatives += 1
    for predicted in predictedList:
        if predicted not in goldList:
            falsePositives += 1
    return EvaluationCounts(truePositives=truePositives, falsePositives=falsePositives, falseNegatives=falseNegatives)


if __name__ == '__main__':

    lines = read_list('./sample_el_q_result.txt')
    q_to_system_answer_dict = dict()
    for line in lines:
        cols = line.split('\t')
        q_to_system_answer_dict[cols[1]] = eval(cols[3])

    gold_lines = read_list('./sample_gold_q_result.txt')
    q_to_gold_answer_dict = dict()
    for gold_line in gold_lines:
        cols = gold_line.split('\t')
        q_to_gold_answer_dict[cols[0]] = eval(cols[1])

    counts = []
    for q, goldList in q_to_gold_answer_dict.items():
        predictedList = []
        if q in q_to_system_answer_dict:
            predictedList = q_to_system_answer_dict[q]
        count = calculate(goldList=goldList, predictedList=predictedList)
        counts.append(count)
    qald_f1 = calculate_f1_qald(counts=counts)
    print (qald_f1)


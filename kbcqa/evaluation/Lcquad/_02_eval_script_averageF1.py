
import json

if __name__ == '__main__':
    averageRecall = 0
    averagePrecision = 0
    averageF1 = 0
    count = 0
    prediction_file_path = './sparqa_results/2021.02.05_output_LC_IR_5_E2E_withnames_all_nonull【定稿】.json'
    with open(prediction_file_path) as prediction_file:
        predictions = json.load(prediction_file)
        for index, prediction in enumerate(predictions):
            f1 = prediction['f1_score']
            recall = prediction['recall_score']
            precision = prediction['precision_score']
            averageRecall += recall
            averagePrecision += precision
            averageF1 += f1
            count += 1

    count = 1000
    """Print final results"""
    averageRecall = float(averageRecall) / count
    averagePrecision = float(averagePrecision) / count
    averageF1 = float(averageF1) / count
    returnString = ""
    returnString += "Number of questions: " + str(count)
    returnString += "\n"
    returnString += "Average recall over questions: " + str(averageRecall)
    returnString += "\n"
    returnString += "Average precision over questions: " + str(averagePrecision)
    returnString += "\n"
    returnString += "Average f1 over questions (accuracy): " + str(averageF1)
    returnString += "\n"
    if averagePrecision + averageRecall > 0:
        averageNewF1 = 2 * averageRecall * averagePrecision / (averagePrecision + averageRecall)
    else:
        averageNewF1 = 0.0
    returnString += "F1 of average recall and average precision: " + str(averageNewF1)
    print(returnString)


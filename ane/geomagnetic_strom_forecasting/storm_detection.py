import numpy
import pandas
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def detect_storm(path_save_file):
    threshold = -50
    pred_results = numpy.array(pandas.read_excel(path_save_file + '/preds.xlsx'))
    preds = numpy.zeros(pred_results.shape[0])
    targets = numpy.zeros(pred_results.shape[0])

    for i in range(0, pred_results.shape[0]):
        if pred_results[i, 0] < threshold:
            targets[i] = 1

        if pred_results[i, 1] < threshold:
            preds[i] = 1

    FP = 0
    FN = 0

    for i in range(0, pred_results.shape[0]):
        if targets[i] == 1 and preds[i] == 0:
            FN += 1

        if targets[i] == 0 and preds[i] == 1:
            FP += 1

    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    f1 = f1_score(targets, preds)

    return precision, recall, f1

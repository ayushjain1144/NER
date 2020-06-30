from sklearn import metrics
import numpy as np

def print_statistics(y_pred, y):
    y_encode = np.argmax(y, axis=0)
    y_pred_encode = np.argmax(y_pred, axis=0)

    print(metrics.classification_report(y_encode, y_pred_encode, digits=10))

    return metrics.classification_report(y_encode, y_pred_encode, digits=10)

def calculate_statistics(y_pred, y):
    y_encode = np.argmax(y, axis=0)
    y_pred_encode = np.argmax(y_pred, axis=0)
    
    f1, prec, rec = metrics.f1_score(y_encode, y_pred_encode, average="macro"), metrics.precision_score(y_encode, y_pred_encode, average="macro"), metrics.recall_score(y_encode, y_pred_encode, average="macro")
    
    return f1, prec, rec

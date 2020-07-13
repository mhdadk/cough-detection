from sklearn.metrics import confusion_matrix

def compute_metrics(y_true, y_pred):

    cfm = confusion_matrix(y_true,y_pred)
    TP = cfm[1,1] # true positive
    TN = cfm[0,0] # true negative
    FP = cfm[0,1] # false positive
    FN = cfm[1,0] # false negative
    sensitivity = TP/(TP+FN) # true positive rate (TPR)
    specificity = TN/(TN+FP) # true negative rate (TNR)
    PPV = TP/(TP+FP) # positive predictive value (or precision)
    NPV = TN/(TN+FN) # negative predictive value
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    balanced_accuracy = (sensitivity+specificity)/2
    MCC = (TP*TN - FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5) # Matthews correlation coefficient
    
    metrics = {'cfm':cfm,
               'sens':sensitivity,
               'spec':specificity,
               'PPV':PPV,
               'NPV':NPV,
               'acc':accuracy,
               'bal_acc':balanced_accuracy,
               'MCC':MCC}
    
    return metrics

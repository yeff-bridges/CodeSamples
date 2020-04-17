import numpy as np
from matplotlib import pyplot as plt

def TruePositive(y, y_hat):
    '''
    :param y: y-values that were used to train model
    :param y_hat: y-values predicted by the model
    :return: True Positive - number of times the model correctly predicted positive
    '''
    return np.sum(y & y_hat)

def FalsePositive(y, y_hat):
    '''
    :param y: y-values that were used to train model
    :param y_hat: y-values predicted by the model
    :return: False Positive - number of times the model incorrectly predicted positive
    '''
    return np.sum(~y & y_hat) #(~) is the bitwise Not operator

def TrueNegative(y, y_hat):
    '''
    :param y: y-values that were used to train model
    :param y_hat: y-values predicted by the model
    :return: True Negative - number of times the model correctly predicted negative
    '''
    return np.sum(~y & ~y_hat)

def FalseNegative(y, y_hat):
    '''
    :param y: y-values that were used to train model
    :param y_hat: y-values predicted by the model
    :return: False Negative - number of times the model incorrectly predicted negative
    '''
    return np.sum(y & ~y_hat)

def Accuracy(y, p_hat, t=0.5):
    '''
    :param y: y values used to train model
    :param p_hat: probabilities
    :param t: threshold
    :return: Accuracy - percentage of correctly predicted values out of all the whole data set
    '''
    y_hat = p_hat >= t
    y = y == 1
    tp = TruePositive(y, y_hat)
    tn = TrueNegative(y, y_hat)
    return (tp + tn) / y.shape[0]

def Precision(y, p_hat, t=0.5):
    '''
    :param y: y values used to train model
    :param p_hat: probabilities
    :param t: threshold
    :return: Precision - correct predictions out of all predictions
    Precision tells us how useful the results are
    '''
    y_hat = p_hat >= t
    y = y == 1
    tp = TruePositive(y, y_hat)
    fp = FalsePositive(y, y_hat)
    return tp / (tp + fp)

def Recall(y, p_hat, t=0.5):
    '''
    :param y: y values used to train model
    :param p_hat: probabilities
    :param t: threshold
    :return: Recall - correct predictions out of all positive instances (not just those predicted)
    Recall tells us how complete the results are
    '''
    y_hat = p_hat >= t
    y = y == 1
    tp = TruePositive(y, y_hat)
    fn = FalseNegative(y, y_hat)
    return tp / (tp + fn)

def F1(y, p_hat, t=0.5):
    '''
    :param y: y values used to train model
    :param p_hat: probabilities
    :param t: threshold
    :return: F1 - a measure of the tests' accuracy; is a harmonic mean of precison and recall
    The test is at it's best when the F1 = 1, and at it's worst when F1 = 0
    '''
    precision = Precision(y, p_hat, t)
    recall = Recall(y, p_hat, t)
    f1 = 2 * precision * recall / (precision + recall)
    if np.isnan(f1):
        return 0
    else:
        return f1

def TPR(y, p_hat, t=0.5):
    '''
    :param y: y values used to train model
    :param p_hat: probabilities
    :param t: threshold
    :return: True Positive Rate - percentage of positives that are correctly identified
    '''
    y_hat = p_hat >= t
    y = y == 1
    tp = TruePositive(y, y_hat)
    fn = FalseNegative(y, y_hat)
    return tp / (tp + fn)

def FPR(y, p_hat, t=0.5):
    '''
    :param y: y values used to train model
    :param p_hat: probabilities
    :param t: threshold
    :return: False Positive Rate - percentage of positives that are incorrectly identified
    '''
    y_hat = p_hat >= t
    y = y == 1
    fp = FalsePositive(y, y_hat)
    tn = TrueNegative(y, y_hat)
    #     print(fp)
    return fp / (fp + tn)

def ROC(y, p_hat, num = 1000):
    '''
    :param y: y values used to train model
    :param p_hat: probabilities
    :param num: number of iterations
    :return: plot of the ROC Curve, includes best F1 and Accuracy scores and AUC

    The ROC curve is performance measurement for classification problem, how much the model is
    capable of distinguishing between classes/features. FPR v. TPR plotted.

    AUC (area under the curve), model is best when AUC = 1, worst when 0.5 (no class separation at all).
    If AUC is negative number then the model is reciprocating (predicting 1 as 0 and vice-versa)
    '''
    tpr = []
    fpr = []
    f1 = []
    accuracy = []

    thresholds = np.linspace(1, 0, num=num) #want to keep track of these to calculate best scores
    for t in thresholds:
        #         print(t)
        tpr.append(TPR(y, p_hat, t))
        fpr.append(FPR(y, p_hat, t))
        f1.append(F1(y, p_hat, t))
        accuracy.append(Accuracy(y, p_hat, t))

    opt_f1 = np.argmax(f1)
    opt_acc = np.argmax(accuracy)

    plt.figure(figsize = (10,6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k:')
    plt.scatter(fpr[opt_f1], [tpr[opt_f1]], c = 'g', marker = 'o', s = 50)
    plt.scatter(fpr[opt_acc], [tpr[opt_acc]], c = 'r', marker = '+', s = 100)


    tpr = np.array(tpr)

    AUC = np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2 )

    print("Best F1 Score:         {:.3} at {:.3}".format(f1[opt_f1], thresholds[opt_f1]))
    print("Best Accuracy Score:   {:.3} at {:.3}".format(f1[opt_acc], thresholds[opt_acc]))
    print("Area Under the Curve:  {:.3}".format(AUC))

    return AUC


def R2(y, y_hat):
    '''
    :param y: y-values that were used to train model
    :param y_hat: y-values predicted by the model
    :return: R-Squared value - a measure of fit - how close the data are to the fitted regression line
    Model is best when R2 = 1 and worst when R2 = 0
    '''
    y_bar = np.mean(y)
    top = np.sum((y - y_hat) ** 2)
    bot = np.sum((y - y_bar) ** 2)
    return 1 - top /bot

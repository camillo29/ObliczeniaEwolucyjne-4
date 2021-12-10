from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import random
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import Perceptron

def Parameters(numberFeatures,icls):
    genome = list()
    #penalty{‘l2’,’l1’,’elasticnet’}
    listPenalty = ["l2", "l1", "elasticnet"]
    genome.append(listPenalty[random.randint(0, 2)])
    #alpha
    a = random.uniform(0.0001, 0.01)
    genome.append(a)
    return icls(genome)



def ParametersFeatures(numberFeatures,icls):
     genome = list()
     # penalty{‘l2’,’l1’,’elasticnet’}
     listPenalty = ["l2", "l1", "elasticnet"]
     genome.append(listPenalty[random.randint(0, 2)])
     # alpha
     a = random.uniform(0.0001, 0.01)
     genome.append(a)
     for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))
     return icls(genome)


def ParametersFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = model_selection.StratifiedKFold(n_splits=split)
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)
    estimator = Perceptron(penalty=individual[0],alpha=individual[1], random_state=101)
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum = resultSum + result
    return resultSum / split,



def ParametersFeatureFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = model_selection.StratifiedKFold(n_splits=split)

    listColumnsToDrop = [] #lista cech do usuniecia
    for i in range(numberOfAtributtes, len(individual)):
        if individual[i] == 0: #gdy atrybut ma zero to usuwamy cechę
            listColumnsToDrop.append(i-numberOfAtributtes)

    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)
    estimator = Perceptron(penalty=individual[0], alpha=individual[1], random_state=101)
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum = resultSum + result
    return resultSum / split,


def mutation(individual):
    numberParamer= random.randint(0, len(individual)-1)
    if numberParamer == 0:
        # penalty{‘l2’,’l1’,’elasticnet’}
        listPenalty = ["l2", "l1", "elasticnet"]
        individual[0] = listPenalty[random.randint(0, 2)]
    elif numberParamer == 1:
        # alpha
        a = random.uniform(0.0001, 0.01)
        individual[1] = a


def mutationFeatures(individual):
    numberParamer= random.randint(0,len(individual)-1)
    if numberParamer == 0:
        # penalty{‘l2’,’l1’,’elasticnet’}
        listPenalty = ["l2", "l1", "elasticnet"]
        individual[0] = listPenalty[random.randint(0, 2)]
    elif numberParamer == 1:
        # alpha
        a = random.uniform(0.0001, 0.01)
        individual[1] = a
    else: #genetyczna selekcja cech
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0
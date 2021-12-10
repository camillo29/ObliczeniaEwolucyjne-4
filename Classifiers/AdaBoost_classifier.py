from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
import random
from sklearn import metrics
from sklearn import model_selection

def Parameters(numberFeatures,icls):
    genome = list()
    #n_estimators
    genome.append(random.randint(20,100))
    #learning_rate
    learning_rate = random.uniform(0.1, 1)
    genome.append(learning_rate)
    #algorithm{‘SAMME’, ‘SAMME.R’}
    algorithmList = ["SAMME", "SAMME.R"]
    genome.append(algorithmList[random.randint(0, 1)])
    return icls(genome)



def ParametersFeatures(numberFeatures,icls):
     genome = list()
     # n_estimators
     genome.append(random.randint(20, 100))
     # learning_rate
     learning_rate = random.uniform(0.1, 1)
     genome.append(learning_rate)
     # algorithm{‘SAMME’, ‘SAMME.R’}
     algorithmList = ["SAMME", "SAMME.R"]
     genome.append(algorithmList[random.randint(0, 1)])
     for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))
     return icls(genome)


def ParametersFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = model_selection.StratifiedKFold(n_splits=split)
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)
    estimator = AdaBoostClassifier(n_estimators=individual[0],learning_rate=individual[1],algorithm=individual[2], random_state=101)
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
    estimator = AdaBoostClassifier(n_estimators=individual[0],learning_rate=individual[1],algorithm=individual[2], random_state=101)
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
        # n_estimators
        individual[0] = random.randint(20, 100)
    elif numberParamer == 1:
        # learning_rate
        individual[1] = random.uniform(0.1, 1)
    elif numberParamer == 2:
        # algorithm{‘SAMME’, ‘SAMME.R’}
        algorithmList = ["SAMME", "SAMME.R"]
        individual[2] = algorithmList[random.randint(0, 1)]


def mutationFeatures(individual):
    numberParamer= random.randint(0,len(individual)-1)
    if numberParamer == 0:
        # n_estimators
        individual[0] = random.randint(20, 100)
    elif numberParamer == 1:
        # learning_rate
        individual[1] = random.uniform(0.1, 1)
    elif numberParamer == 2:
        # algorithm{‘SAMME’, ‘SAMME.R’}
        algorithmList = ["SAMME", "SAMME.R"]
        individual[2] = algorithmList[random.randint(0, 1)]
    else: #genetyczna selekcja cech
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0
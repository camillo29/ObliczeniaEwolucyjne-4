from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import random
from sklearn import metrics
from sklearn import model_selection

def Parameters(numberFeatures,icls):
    genome = list()
    #acvitation
    listActivation = ["identity", "logistic", "tanh", "relu"]
    genome.append(listActivation[random.randint(0, 3)])
    #solver
    listSolver = ["lbfgs", "sgd", "adam"]
    genome.append(listSolver[random.randint(0,2)])
    #alpha
    genome.append(random.uniform(0.001, 5))
    # shuffle
    shuffle = random.choice([True, False])
    genome.append(shuffle)
    return icls(genome)



def ParametersFeatures(numberFeatures,icls):
     genome = list()
     # acvitation
     listActivation = ["identity", "logistic", "tanh", "relu"]
     genome.append(listActivation[random.randint(0, 3)])
     # solver
     listSolver = ["lbfgs", "sgd", "adam"]
     genome.append(listSolver[random.randint(0, 2)])
     # alpha
     genome.append(random.uniform(0.001, 5))
     # shuffle
     shuffle = random.choice([True, False])
     genome.append(shuffle)
     for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))
     return icls(genome)


def ParametersFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = model_selection.StratifiedKFold(n_splits=split)
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)
    estimator = MLPClassifier(activation=individual[0], solver=individual[1], alpha=individual[2],
                                max_iter=1500)
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
    estimator = MLPClassifier(activation=individual[0], solver=individual[1], alpha=individual[2],
                                 max_iter=2000)
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
        # acvitation
        listActivation = ["identity", "logistic", "tanh", "relu"]
        individual[0] = listActivation[random.randint(0, 3)]
    elif numberParamer == 1:
        # solver
        listSolver = ["lbfgs", "sgd", "adam"]
        individual[1] = listSolver[random.randint(0, 2)]
    elif numberParamer == 2:
        # shuffle
        shuffle = random.choice([True, False])
        individual[2] = shuffle


def mutationFeatures(individual):
    numberParamer= random.randint(0,len(individual)-1)
    if numberParamer == 0:
        # acvitation
        listActivation = ["identity", "logistic", "tanh", "relu"]
        individual[0] = listActivation[random.randint(0, 3)]
    elif numberParamer == 1:
        # solver
        listSolver = ["lbfgs", "sgd", "adam"]
        individual[1] = listSolver[random.randint(0, 2)]
    elif numberParamer == 2:
        # shuffle
        shuffle = random.choice([True, False])
        individual[2] = shuffle
    else: #genetyczna selekcja cech
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0
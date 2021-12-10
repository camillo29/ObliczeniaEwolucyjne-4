from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn import metrics
from sklearn import model_selection

def Parameters(numberFeatures,icls):
    genome = list()
    #Neighbours
    n_neighbours = random.randint(5, 20)
    genome.append(n_neighbours)
    #Weights
    listWeights = ["uniform", "distance"]
    genome.append(listWeights[random.randint(0, 1)])
    #Algorithm
    listAlgorithm = ["auto", "ball_tree", "kd_tree", "brute"]
    genome.append(listAlgorithm[random.randint(0, 3)])
    #Leaf size
    leafSize = random.randint(20, 50)
    genome.append(leafSize)
    return icls(genome)



def ParametersFeatures(numberFeatures,icls):
     genome = list()
     # Neighbours
     n_neighbours = random.randint(5, 20)
     genome.append(n_neighbours)
     # Weights
     listWeights = ["uniform", "distance"]
     genome.append(listWeights[random.randint(0, 1)])
     # Algorithm
     listAlgorithm = ["auto", "ball_tree", "kd_tree", "brute"]
     genome.append(listAlgorithm[random.randint(0, 3)])
     # Leaf size
     leafSize = random.randint(20, 50)
     genome.append(leafSize)
     for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))
     return icls(genome)


def ParametersFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = model_selection.StratifiedKFold(n_splits=split)
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)
    estimator = KNeighborsClassifier(n_neighbors=individual[0], weights=individual[1], algorithm=individual[2], leaf_size=individual[3])
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
    estimator = KNeighborsClassifier(n_neighbors=individual[0], weights=individual[1], algorithm=individual[2], leaf_size=individual[3])
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
    numberParamer = random.randint(0, len(individual)-1)
    if numberParamer == 0:
        # kn_neighbours
        n_neighbours = random.randint(5, 20)
        individual[0] = n_neighbours
    elif numberParamer == 1:
        # Weights
        listWeights = ["uniform", "distance"]
        individual[1] = listWeights[random.randint(0, 1)]
    elif numberParamer == 2:
        # Algorithm
        listAlgorithm = ["auto", "ball_tree", "kd_tree", "brute"]
        individual[2] = listAlgorithm[random.randint(0, 3)]
    elif numberParamer == 3:
        #gamma
        leafSize = random.randint(20, 50)
        individual[3] = leafSize


def mutationFeatures(individual):
    numberParamer = random.randint(0, len(individual)-1)
    if numberParamer == 0:
        # kn_neighbours
        n_neighbours = random.randint(5, 20)
        individual[0] = n_neighbours
    elif numberParamer == 1:
        # Weights
        listWeights = ["uniform", "distance"]
        individual[1] = listWeights[random.randint(0, 1)]
    elif numberParamer == 2:
        # Algorithm
        listAlgorithm = ["auto", "ball_tree", "kd_tree", "brute"]
        individual[2] = listAlgorithm[random.randint(0, 3)]
    elif numberParamer == 3:
        # gamma
        leafSize = random.randint(20, 50)
        individual[3] = leafSize
    else: #genetyczna selekcja cech
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import random
from sklearn import metrics
from sklearn import model_selection

def Parameters(numberFeatures,icls):
    genome = list()
    #Criterion
    listCriterion = ["gini", "entropy"]
    genome.append(listCriterion[random.randint(0, 1)])
    #Spliter
    listSpliter = ["best", "random"]
    genome.append(listSpliter[random.randint(0, 1)])
    #max_depth
    max_depth = random.randint(3, 10)
    genome.append(max_depth)
    #min_samples_split
    min_samples_split = random.randint(2, 10)
    genome.append(min_samples_split)
    # min_samples_leaf
    min_samples_leaf = random.randint(1, 5)
    genome.append(min_samples_leaf)
    # max_features
    listMaxFeatures = ["auto", "sqrt", "log2"]
    genome.append(listMaxFeatures[random.randint(0, 2)])
    return icls(genome)



def ParametersFeatures(numberFeatures,icls):
     genome = list()
     # Criterion
     listCriterion = ["gini", "entropy"]
     genome.append(listCriterion[random.randint(0, 1)])
     # Splitter
     listSplitter = ["best", "random"]
     genome.append(listSplitter[random.randint(0, 1)])
     # max_depth
     max_depth = random.randint(3, 10)
     genome.append(max_depth)
     # min_samples_split
     min_samples_split = random.randint(2, 10)
     genome.append(min_samples_split)
     # min_samples_leaf
     min_samples_leaf = random.randint(1, 5)
     genome.append(min_samples_leaf)
     # max_features
     listMaxFeatures = ["auto", "sqrt", "log2"]
     genome.append(listMaxFeatures[random.randint(0, 2)])
     for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))
     return icls(genome)


def ParametersFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = model_selection.StratifiedKFold(n_splits=split)
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)
    estimator = DecisionTreeClassifier(criterion=individual[0],splitter=individual[1],max_depth=individual[2], min_samples_split=individual[3],min_samples_leaf=individual[4], max_features=individual[5], random_state=101)
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
    estimator = DecisionTreeClassifier(criterion=individual[0],splitter=individual[1],max_depth=individual[2], min_samples_split=individual[3],min_samples_leaf=individual[4], max_features=individual[5], random_state=101)
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
        # Criterion
        listCriterion = ["gini", "entropy"]
        individual[0] = listCriterion[random.randint(0, 1)]
    elif numberParamer == 1:
        # Splitter
        listSplitter = ["best", "random"]
        individual[1] = listSplitter[random.randint(0, 1)]
    elif numberParamer == 2:
        # max_depth
        max_depth = random.randint(3, 10)
        individual[2] = max_depth
    elif numberParamer == 3:
        # min_samples_split
        min_samples_split = random.randint(2, 10)
        individual[3] = min_samples_split
    elif numberParamer == 4:
        # min_samples_leaf
        min_samples_leaf = random.randint(1, 5)
        individual[4] = min_samples_leaf
    elif numberParamer == 5:
        # max_features
        listMaxFeatures = ["auto", "sqrt", "log2"]
        individual[5] = listMaxFeatures[random.randint(0, 2)]


def mutationFeatures(individual):
    numberParamer= random.randint(0,len(individual)-1)
    if numberParamer == 0:
        # Criterion
        listCriterion = ["gini", "entropy"]
        individual[0] = listCriterion[random.randint(0, 1)]
    elif numberParamer == 1:
        # Splitter
        listSplitter = ["best", "random"]
        individual[1] = listSplitter[random.randint(0, 1)]
    elif numberParamer == 2:
        # max_depth
        max_depth = random.randint(3, 10)
        individual[2] = max_depth
    elif numberParamer == 3:
        # min_samples_split
        min_samples_split = random.randint(2, 10)
        individual[3] = min_samples_split
    elif numberParamer == 4:
        # min_samples_leaf
        min_samples_leaf = random.randint(1, 5)
        individual[4] = min_samples_leaf
    elif numberParamer == 5:
        # max_features
        listMaxFeatures = ["auto", "sqrt", "log2"]
        individual[5] = listMaxFeatures[random.randint(0, 2)]
    else: #genetyczna selekcja cech
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0
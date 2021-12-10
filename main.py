import pandas as pd
import numpy as np
import multiprocessing
import random
import math

from deap import base
from deap import creator
from deap import tools

from scipy.io.arff import loadarff

from sklearn import metrics
from sklearn import model_selection
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron

from Classifiers import SVC_classifier
from Classifiers import NearestNeighbours_classifier
from Classifiers import Tree_classifier
from Classifiers import MLP_classifier
from Classifiers import AdaBoost_classifier as AB_classifier
from Classifiers import RandomForest_classifier
from Classifiers import Perceptron_classifier




def loadData(dataSet):
    pd.set_option('display.max_columns', None)
    if dataSet == "data/heart_failure_clinical_records_dataset.csv":
        df = pd.read_csv("data/heart_failure_clinical_records_dataset.csv", sep=',')
        y = df['DEATH_EVENT']
        df.drop('DEATH_EVENT', axis=1, inplace=True)
    elif dataSet == "data/data.csv":
        df = pd.read_csv("data/data.csv", sep=',')
        y = df['Status']
        df.drop('Status', axis=1, inplace=True)
        df.drop('ID', axis=1, inplace=True)
        df.drop('Recording', axis=1, inplace=True)
    return df, y


def app():

    # Original
    #df, y = loadData("data/data.csv")
    # Project
    df, y = loadData("data/heart_failure_clinical_records_dataset.csv")
    df.columns = range(df.shape[1])
    print(df)
    numberOfAtributtes = len(df.columns)
    print("n of atr = ", numberOfAtributtes)
    print("len(df) = ", len(df))
    print("df head\n", df.head())
    print("y.head\n ", y)
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    #clf = SVC()
    #clf = KNeighborsClassifier()
    #clf = DecisionTreeClassifier()
    #clf = MLPClassifier(max_iter=1500) #Requires a lot of time to compute but it works
    #clf = AdaBoostClassifier()
    #clf = RandomForestClassifier()
    clf = Perceptron()
    scores = model_selection.cross_val_score(clf, df_norm, y,
                                             cv=5, scoring='accuracy',
                                             n_jobs=-1,
                                             error_score='raise')
    print(scores.mean())
    mainLoop(df, y)

def createToolBox(df, y, classifier):
    toolbox = base.Toolbox()
    toolbox.register('individual', classifier.ParametersFeatures, len(df.columns),
                     creator.Individual)
    toolbox.register("evaluate", classifier.ParametersFeatureFitness, y, df, len(df.columns))
    toolbox.register("mutate", classifier.mutationFeatures)
    return toolbox


def mainLoop(df, y):
    probabilityMutation = 0.8
    probabilityCrossover = 0.2
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = createToolBox(df, y, Perceptron_classifier)

    toolbox.register("select", tools.selTournament, tournsize = 3)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(n=100)
    g = 0
    numberElitism = 0
    numberIteration = 100
    prevBest = tools.selBest(pop, 1)[0]
    while g < numberIteration:
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        listElitism = []
        for x in range(0, numberElitism):
            listElitism.append(tools.selBest(pop, 1)[0])

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < probabilityCrossover:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < probabilityMutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print(" Evaluated %i individuals" % len(invalid_ind))
        pop[:] = offspring + listElitism
        if(tools.selBest(pop, 1)[0].fitness.values>prevBest.fitness.values):
            prevBest = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (prevBest,
                                             prevBest.fitness.values))


if __name__ == '__main__':
    app()


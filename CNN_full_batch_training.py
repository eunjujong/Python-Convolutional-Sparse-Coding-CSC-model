from baseClass_CNN_2 import InputLayer, ConvLayer, PoolLayer_Max, FlattenLayer, ReLuLayer, MLP
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from PIL import Image 
import os
import math 
 
class CNN: 
 
    def __init__(self, num_filters, kernel_size, meanX, stdX, stride, padding): 
        self.inputLayer = InputLayer(meanX, stdX) 
        self.convolutionLayer = ConvLayer(num_filters, kernel_size, stride, padding) 
        self.maxPoolLayer = PoolLayer_Max(num_filters, stride)
        self.reLuLayer = ReLuLayer()
        self.convolutionLayer2 = ConvLayer(num_filters, kernel_size, stride, padding) 
        self.maxPoolLayer2 = PoolLayer_Max(num_filters, stride)
        self.reLuLayer2 = ReLuLayer()
        # self.reLuLayer2 = ReLuLayer()
        self.listOfLayers =  [self.convolutionLayer, self.maxPoolLayer, self.convolutionLayer2, self.maxPoolLayer2]
        
    
    def forwardPropagate(self, X): 
 
        outputConvolution = self.convolutionLayer.forward_propagate(X)
 
        outputMaxPool1 = self.maxPoolLayer.forward_propagate(outputConvolution) 
 
        outputConv2 = self.convolutionLayer2.forward_propagate(outputMaxPool1) 
 
        outputMaxPool2 = self.maxPoolLayer2.forward_propagate(outputConv2) 
 
        return outputMaxPool2
 
    def backwardPropagate(self, gradient): 
 
        gradientObj = gradient 
 
        m = len(self.listOfLayers)
 
        for i in range(m - 1, -1, -1):
            gradientObj = self.listOfLayers[i].backwardsPropagate(gradientObj) 
 
 
        return gradientObj
 
class MedicalClassifier: 
 
    def __init__(self, num_filters, kernel_size, pool_kernel_size, stride, padding, numberofFCLayers, numberOfHiddenLayers, learning_rate, sizes, meanX, stdX, lambdaScalar): 
        self.CNN = CNN(num_filters, kernel_size, meanX, stdX, stride, padding)
        self.FlattenLayer = FlattenLayer()
        self.lr = learning_rate
        self.MLP = MLP(numberofFCLayers, numberOfHiddenLayers, sizes, self.lr, lambdaValue=lambdaScalar)
 
 
    def forwardPropagate(self, X, Y):
 
        X = np.transpose(X, (0, 2, 3, 1)) 
 
        outputConvolution = self.CNN.forwardPropagate(X)
 
        flattenInput = self.FlattenLayer.forwardPropagate(outputConvolution) 
 
        outputMatrix, output = self.MLP.forwardPropagate(flattenInput, Y)
 
 
        return outputMatrix, output
 
    def backwardPropagate(self): 
        gradientObj = self.MLP.backwardPropagate() 
 
        gradientObj = self.FlattenLayer.backwardsPropagate(gradientObj) 
 
        gradientObj = self.CNN.backwardPropagate(gradientObj)
 
        return gradientObj
 
# Citation: YXD/dsalaj on stackoverflow
# https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
# Convert Array of Indices To 1 Hot Encoded Numpy Array
def one_hot(a): 
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size), a] = 1 
    return b
 
# Citation: DSM on stackoverflow
# https://stackoverflow.com/questions/20295046/numpy-change-max-in-each-row-to-1-all-other-numbers-to-0
# Numpy: change max in each row to 1, all other numbers to 0
def convertToOne_hot(a): 
    b = np.zeros_like(a)
    b[np.arange(len(a)), a.argmax(1)] = 1
    return b
    
if __name__ == '__main__': 
    
    trainingSet = pd.read_csv('train-rle.csv', sep=',', header=None)

    trainingSet = trainingSet.sample(frac=1).reset_index(drop=True)

    trainingSet = trainingSet.to_numpy() 

    trainingSetNames = trainingSet[np.s_[:8462], np.s_[0]]

    trainingSetLabels = trainingSet[np.s_[:8462], np.s_[1]]

    validationSetNames = trainingSet[np.s_[8462:], np.s_[0]]

    validationSetLabels = trainingSet[np.s_[8462:], np.s_[1]]

    outputSizes = [15376, 2500, 1250, 2]

    logLoss_trainingList, logLoss_validationList = [0],  [0]

    learning_rate = 1e-4

    validation_set = None 

    validationSet_labels = None 

    training_set = None 

    trainingSet_labels = None
    
    nextImage = None

    nextImage2 = None

    training_set_index = None

    validation_set = np.load('validation_set.npy')

    print(validation_set.shape)

    validationSet_labels = np.load('validationSet_labels.npy')

    preservedValidationLabels = validationSet_labels

    validationSet_labels = one_hot(validationSet_labels) 

    # dataset = np.load('training_set.npy')

    # dataset_labels = np.load('trainingSet_labels.npy')

    # print("BEFORE", dataset_labels.shape)

    # preservedDatasetLabels  = dataset_labels

    # dataset_labels = one_hot(dataset_labels)

    # print("NOW", dataset_labels.shape)

    # dataset = dataset / 255.0

    medicalClassifier = MedicalClassifier(1, 5, 5, 1, 1, 3,2, learning_rate, outputSizes, 0, 255, 0)
 
    outputTrainingMatrix = None
 
    trainAccuracies = []
 
    testAccuracies = []
 
    trainLosses = []

    train_truePositive, train_trueNegative, train_falsePositive, train_falseNegative = 0, 0, 0, 0

    specificityAccuracy, sensitivityAccuracy = [], [] 
    
    # trueLabels = np.argwhere(preservedDatasetLabels == 1) 

    # nonTrueLabels = np.argwhere(preservedDatasetLabels == 0) 

    # nonTrueDataset = dataset[nonTrueLabels] 

    # trueDataset = dataset[trueLabels]
    # print(trueLabels) 
    # print(nonTrueLabels)
    # print("NON TRUE DATASET", preservedDatasetLabels[nonTrueLabels]) 
    # print("TRUE DATASET", preservedDatasetLabels[trueLabels])

    # i = 0
    
    # trueDatasetLabels = preservedDatasetLabels[trueLabels]

    # nonTrueDatasetLabels = preservedDatasetLabels[nonTrueLabels]
    
    # while nonTrueDataset.shape[0] != trueDataset.shape[0]: 
        
    #     trueExample = np.expand_dims(trueDataset[i], axis=0)

    #     trueDataset = np.append(trueDataset, trueExample, axis=0)
        
    #     newTrueLabel = np.expand_dims(trueDatasetLabels[i], axis=0) 

    #     trueDatasetLabels = np.append(trueDatasetLabels, newTrueLabel, axis=0)
        
    #     i += 1

    # print("FALSE DATASET", nonTrueDataset.shape[0]) 
    # print("TRUE DATASET", trueDataset.shape[0])

    # balancedDataset = np.append(trueDataset, nonTrueDataset, axis=0) 
    
    # balancedDatasetLabels = np.append(trueDatasetLabels, nonTrueDatasetLabels, axis=0)
    # print(balancedDatasetLabels)
    # balancedDatasetLabels = np.squeeze(balancedDatasetLabels, axis=1)
    # balancedDatasetLabelsIndex = balancedDatasetLabels 
    # print(balancedDatasetLabels.shape)
    # balancedDatasetLabels = one_hot(balancedDatasetLabelsIndex) 
    # print("FULL BALANCED DATASET LABELS", balancedDatasetLabels)
    # print("FULL BALANCED DATASET", balancedDataset.shape)
    # print("FULL BALANCED DATASET LABELS", balancedDatasetLabels.shape)
    # print("FULL BALANCED balancedDataset index", balancedDatasetLabelsIndex.shape)
    # balancedDataset = np.squeeze(balancedDataset, axis=1) 
    # with open('balancedDataset.npy', 'wb') as f: 
    #         np.save(f, balancedDataset) 

    # with open('balancedDatasetLabels.npy', 'wb') as f: 
    #        np.save(f, balancedDatasetLabels) 
    
    balancedDataset = np.load('balancedDataset.npy') 

    balancedDatasetLabels = np.load('balancedDatasetLabels.npy') 
    
    balancedDatasetLabelsIndex = np.argmax(balancedDatasetLabels, axis=1) 

    number_of_training_rows = balancedDataset.shape[0]
    
    shuffler = np.random.permutation(balancedDataset.shape[0])
    balancedDataset = balancedDataset[shuffler]
    balancedDatasetLabels = balancedDatasetLabels[shuffler]
    balancedDatasetLabelsIndex = balancedDatasetLabelsIndex[shuffler] 

    training_set = balancedDataset

    trainingSet_labels = balancedDatasetLabels

    nextImage = None

    number_of_rows = balancedDataset.shape[0]

    number_of_validation_rows =  validation_set.shape[0]

    batch_size = 2000 

    epochs = 100

    for i in range(epochs): 

        i = 0

        while i <= number_of_rows/batch_size: 

            randomIndices =  np.random.choice(number_of_rows, size=batch_size, replace=False)

            miniTrainingSet = training_set[randomIndices, :]

            miniTrainingSetLabels = trainingSet_labels[randomIndices]

            print(miniTrainingSet.shape)
                    
            outputMatrix, trainingOutput = medicalClassifier.forwardPropagate(miniTrainingSet, miniTrainingSetLabels)
                        
            medicalClassifier.backwardPropagate() 

            randomIndices =  np.random.choice(number_of_validation_rows, size=batch_size, replace=False)

            miniValidationSet = validation_set[randomIndices, :]

            miniValidationLabels = validationSet_labels[randomIndices]

            outputValidationMatrix, validationOutput = medicalClassifier.forwardPropagate(miniValidationSet, miniValidationLabels)

            print("TRAINING LOSS", trainingOutput) 

            print("VALIDATION LOSS", validationOutput)

            logLoss_trainingList.append(trainingOutput) 

            logLoss_validationList.append(validationOutput) 

            nextImage = np.argmin(np.min(outputMatrix, axis=1), axis=0)

            max_arr_training = outputMatrix 

            max_arr_validation = outputValidationMatrix

            i += 1 

        
    max_arr_training = np.argmax(max_arr_training, axis=1)

    trainingCorrect = np.sum(max_arr_training == trainingSetLabels)

    trainingTotal = trainingSetLabels.shape[0]

    trainingAccuracy = trainingCorrect / trainingTotal 

    print("training accuracy", trainingAccuracy)
            
    max_arr_validation = np.argmax(max_arr_validation, axis=1)

    validationCorrect = np.sum(max_arr_validation == validationSetLabels)

    validationTotal = validationSetLabels.shape[0]

    validationAccuracy = validationCorrect / validationTotal 

    print("validation accuracy", validationAccuracy)
    
    logLoss_trainingList[i].pop(0)
    logLoss_validationList[i].pop(0)

    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(logLoss_trainingList,label="Training Log Loss")
    plt.plot(logLoss_validationList,label="Validation Log Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()   
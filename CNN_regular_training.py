from baseClass_CNN import InputLayer, ConvolutionLayer, MaxPoolingLayer, FlattenLayer, ReLuLayer, MLP
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from PIL import Image 
import os

class CNN: 

    def __init__(self, num_filters, kernel_size, pool_kernel_size, meanX, stdX, stride, padding): 
        self.inputLayer = InputLayer(meanX, stdX) 
        self.convolutionLayer = ConvolutionLayer(num_filters, kernel_size, stride, padding) 
        self.maxPoolLayer = MaxPoolingLayer(num_filters, pool_kernel_size, stride, padding)
        self.convolutionLayer2 = ConvolutionLayer(num_filters, kernel_size, stride, padding) 
        self.maxPoolLayer2 = MaxPoolingLayer(num_filters, pool_kernel_size, stride, padding)
        self.reLuLayer = ReLuLayer()
        self.reLuLayer2 = ReLuLayer()
        self.listOfLayers =  [self.inputLayer, self.convolutionLayer, self.reLuLayer, self.maxPoolLayer,  self.convolutionLayer2, self.reLuLayer2, self.maxPoolLayer2]
        
    
    def forwardPropagate(self, X): 

        outputInputLayer = self.inputLayer.forwardPropagate(X) 

        outputConvolution = self.convolutionLayer.forwardPropagate(outputInputLayer)

        outputReLu = self.reLuLayer.forwardPropagate(outputConvolution)

        outputMaxPool1 = self.maxPoolLayer.forwardPropagate(outputReLu) 

        outputConvolutionLayer2 = self.convolutionLayer2.forwardPropagate(outputMaxPool1) 

        outputReLu2 = self.reLuLayer2.forwardPropagate(outputConvolutionLayer2)


        outputMaxPool2 = self.maxPoolLayer2.forwardPropagate(outputReLu2) 

        return outputMaxPool2

    def backwardPropagate(self, gradient): 

        gradientObj = gradient 

        m = len(self.listOfLayers)

        for i in range(m - 1, 0, -1):
            gradientObj = self.listOfLayers[i].backwardsPropagate(gradientObj) 


        return gradientObj

    def getListOfLayers(self): 

        return self.listOfLayers

class MedicalClassifier: 

    def __init__(self, num_filters, kernel_size, pool_kernel_size, stride, padding, numberofFCLayers, numberOfHiddenLayers, learning_rate, sizes, meanX, stdX): 
        self.CNN = CNN(num_filters, kernel_size, pool_kernel_size, meanX, stdX, stride, padding)
        self.FlattenLayer = FlattenLayer()
        self.lr = learning_rate
        self.MLP = MLP(numberofFCLayers, numberOfHiddenLayers, sizes, self.lr)
        self.listOfLayers = self.CNN.getListOfLayers() 
        self.listOfLayers = self.listOfLayers + [self.FlattenLayer]
        self.listOfLayers = self.listOfLayers + self.MLP.getListOfLayers()
        self.numberOfLayers = len(self.listOfLayers)


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

    def updateWeights(self): 
        for i in range(len(self.listOfLayers)): 
            if hasattr(self.listOfLayers[i], 'weight'): 
                self.listOfLayers[i].updateWeights() 

if __name__ == '__main__': 
    
    trainingSet = pd.read_csv('train-rle.csv', sep=',', header=None)

    trainingSet = trainingSet.sample(frac=1).reset_index(drop=True)

    trainingSet = trainingSet.to_numpy() 

    trainingSetNames = trainingSet[np.s_[:8462], np.s_[0]]

    trainingSetLabels = trainingSet[np.s_[:8462], np.s_[1]]

    validationSetNames = trainingSet[np.s_[8462:], np.s_[0]]

    validationSetLabels = trainingSet[np.s_[8462:], np.s_[1]]

    outputSizes = [441, 360, 290, 2]

    logLoss_trainingList, logLoss_validationList = [0],  [0]

    learning_rate = 1e-3

    medicalClassifier = MedicalClassifier(9, (3,3), (3, 3), 2, 1, 3, 1, learning_rate, outputSizes, 0, 255)

    validation_set = None 

    validationSet_labels = None 

    training_set = None 

    trainingSet_labels = None
    
    validation_set = np.load('validation_set.npy')

    validation_set = np.squeeze(validation_set, axis=1)

    validationSet_labels = np.load('validation_labels.npy')

    dataset = np.load('training_set.npy')

    dataset_labels = np.load('trainingSet_labels.npy')

    validationSet_labels = np.expand_dims(validationSet_labels, axis=1) 

    dataset_labels = np.expand_dims(dataset_labels, axis=1)

    training_set = dataset

    trainingSet_labels = dataset_labels

    print("Training Set ", dataset.shape) 

    print("Training Set Labels", dataset_labels.shape)

    print("Validation Set ", validation_set.shape) 

    print("Validation Set Labels", validationSet_labels.shape)

    nextImage = None

    number_of_rows = dataset.shape[0]

    number_of_validation_rows =  validation_set.shape[0]

    batch_size = 100 

    epochs = 100

    for i in range(epochs): 

        i = 0

        while i <= number_of_rows/batch_size: 

            randomIndices =  np.random.choice(number_of_rows, size=batch_size, replace=False)

            miniTrainingSet = training_set[randomIndices, :]

            miniTrainingSetLabels = trainingSet_labels[randomIndices]

            print(miniTrainingSetLabels.shape, "TrainingSet Labels")

                    
            outputMatrix, trainingOutput = medicalClassifier.forwardPropagate(miniTrainingSet, miniTrainingSetLabels)
                        
            medicalClassifier.backwardPropagate() 

            medicalClassifier.updateWeights() 

            randomIndices =  np.random.choice(number_of_validation_rows, size=batch_size, replace=False)

            miniValidationSet = validation_set[randomIndices, :]

            miniValidationLabels = validationSet_labels[randomIndices]

            print(miniValidationLabels.shape, "Validation Labels")

            outputValidationMatrix, validationOutput = medicalClassifier.forwardPropagate(miniValidationSet, miniValidationLabels)

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
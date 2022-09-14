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
 
    N = 1
 
    trainingSet = trainingSet.iloc[N: , :]
 
    trainingSet = trainingSet.sample(frac=1).reset_index(drop=True)
 
    trainingSet = trainingSet.to_numpy() 
 
    trainingSetNames = trainingSet[np.s_[:8462], np.s_[0]]
 
    trainingSetLabels = trainingSet[np.s_[:8462], np.s_[1]]
 
    validationSetNames = trainingSet[np.s_[8462:], np.s_[0]]
 
    validationSetLabels = trainingSet[np.s_[8462:], np.s_[1]]
 
    outputSizes = [1296, 850, 700, 2]
 
    logLoss_trainingList, logLoss_validationList = [0],  [0]
 
    validation_set = None 
 
    tempvalidationSet_labels = []
 
    # validation_set = None
 
    # validationSet_labels = []
 
    # for i in range(validationSetNames.shape[0]): 
 
    #     abspath = os.path.abspath("./TrainingSet128/" + validationSetNames[i] + ".jpg")
 
    #     trainingInstance = Image.open(abspath)
 
    #     np_image = np.array(trainingInstance)
 
    #     trainingInstance = np.reshape(trainingInstance, (1, 32, 32)) 
 
    #     print("VALIDATION, HERE I AM, ITERATION #{}".format(i))
 
    #     trainingInstance =  np.expand_dims(trainingInstance, axis=0)
 
    #     if i == 0:
 
    #         validation_set = trainingInstance
 
    #         newLabelCheck = validationSetLabels[i]
 
    #         if len(newLabelCheck) == 2: 
    #             validationSet_labels.append(0)
    #         else: 
    #             validationSet_labels.append(1)
 
    #     else: 
 
    #         validation_set = np.append(validation_set, trainingInstance, axis=0) 
 
    #         newLabelCheck = validationSetLabels[i]
 
    #         if len(newLabelCheck) == 2: 
    #             validationSet_labels.append(0)
    #         else: 
    #             validationSet_labels.append(1)
 
    # dataset = None
 
    # dataset_labels = []
 
    # for i in range(trainingSetNames.shape[0]): 
 
    #     abspath = os.path.abspath("./TrainingSet128/" + trainingSetNames[i] + ".jpg")
    #     try: 
    #         trainingInstance = Image.open(abspath)
    #     except FileNotFoundError: 
    #         continue 
 
    #     np_image = np.array(trainingInstance)
 
    #     trainingInstance = np.reshape(trainingInstance, (1, 32, 32)) 
 
    #     print("VALIDATION, HERE I AM, ITERATION #{}".format(i))
 
    #     trainingInstance =  np.expand_dims(trainingInstance, axis=0)
 
    #     if i == 0:
 
    #         balancedDataset = trainingInstance
 
    #         newLabelCheck = trainingSetLabels[i]
 
    #         if len(newLabelCheck) == 2: 
    #             dataset_labels.append(0)
    #         else: 
    #             dataset_labels.append(1)
 
    #     else: 
 
    #         balancedDataset = np.append(balancedDataset, trainingInstance, axis=0) 
 
    #         newLabelCheck = trainingSetLabels[i]
 
    #         if len(newLabelCheck) == 2: 
    #             dataset_labels.append(0)
    #         else: 
    #             dataset_labels.append(1)
 
 
#    nextImage = None
# 
#    nextImage2 = None
#
#    training_set_index = None
# 
#    validation_set = np.load('validation_set.npy')
# 
#    print(validation_set.shape)
# 
#    validationSet_labels = np.load('validationSet_labels.npy')
# 
#    preservedValidationLabels = validationSet_labels
# 
#    validationSet_labels = one_hot(validationSet_labels) 
# 
#    dataset = np.load('training_set.npy')
# 
#    dataset_labels = np.load('trainingSet_labels.npy')
# 
#    print("BEFORE", dataset_labels.shape)
# 
    #preservedDatasetLabels  = np.argmax(dataset_labels,axis=1)
# 
#    print("NOW", dataset_labels.shape)
# 
#    dataset = dataset / 255.0
 
    # with open('validation_set.npy', 'wb') as f: 
    #     np.save(f, validation_set)
 
    # with open('validationSet_Labels.npy', 'wb') as f: 
    #     np.save(f, validationSet_labels)
 
    # with open('training_set.npy', 'wb') as f: 
    #     np.save(f, dataset)
 
    # with open('trainingSet_labels.npy', 'wb') as f: 
    #     np.save(f, dataset_labels)


 
 
    training_set_labels = None
 
    batch_size = 25
 
    randomIndices = [1]
 
    number_of_epochs = 8
    
    learning_rate = 5e-4
 
    lambdaValue = 5.12
 
    medicalClassifier = MedicalClassifier(1, 5, 5, 1, 1, 3,2, learning_rate, outputSizes, 0, 255, lambdaValue)
 
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

    for i in range(25): 
 
        print("EPOCH #{}".format(i))
        
        medicalClassifier = MedicalClassifier(9, 5, 5, 1, 1, 3,2, learning_rate, outputSizes, 0, 255, lambdaValue)

        if i == 0:
 
            randomIndices =  np.random.choice(number_of_training_rows, size=1, replace=False)
            np_image = balancedDataset[randomIndices]
            np_label = balancedDatasetLabels[randomIndices]
 
            training_set = np_image
 
            training_set_labels = np_label

            training_set_index = balancedDatasetLabelsIndex[randomIndices]

        else: 
            
            for i in range(25):
            
                newTrainingInstance = np.expand_dims(balancedDataset[nextImage[i]-i], axis=0)
                
                training_set = np.append(training_set, newTrainingInstance, axis=0) 
 
                np_label = np.expand_dims(balancedDatasetLabels[nextImage[i]-i], axis=0)
                
                training_set_labels = np.append(training_set_labels, np_label, axis=0)
                
                balancedDataset = np.delete(balancedDataset, nextImage[i]-i, axis=0)
 
                balancedDatasetLabels = np.delete(balancedDatasetLabels, nextImage[i]-i, axis=0)

                np_index = np.expand_dims(balancedDatasetLabelsIndex[nextImage[i]-i], axis=0)
                

                training_set_index = np.append(training_set_index, np_index, axis=0)
               
                balancedDatasetLabelsIndex = np.delete(balancedDatasetLabelsIndex, nextImage[i]-i, axis=0) 


#        mask = np.hstack([np.random.choice(np.where(training_set_labels == l)[0], training_set.shape[0], replace=True)
 #                       for l in np.unique(training_set_labels)])
 
  #      tempTrainingSet = training_set[mask]
 
   #     tempTrainingSetLabels = training_set_labels[mask]

    #    print(tempTrainingSetLabels) 
        
        trueLabels = np.argwhere(training_set_index == 1)
        falseLabels = np.argwhere(training_set_index == 0) 
        print(trueLabels.shape[0]) 
        print(falseLabels.shape[0])
        if trueLabels.shape[0] != 0 and falseLabels.shape[0] != 0 and trueLabels.shape[0] < falseLabels.shape[0]: 

            training_true = training_set[trueLabels] 
            training_false = training_set[falseLabels] 

            training_true_labels = training_set_labels[trueLabels] 
            training_false_labels = training_set_labels[falseLabels] 

            print("TRAINING TRUE LABELS", training_true_labels.shape[0]) 
            print("TRAINING FALSE LABELS", training_false_labels.shape[0])
            i = 0

            while training_true.shape[0] != training_false.shape[0]: 
                trueExample = np.expand_dims(training_true[i], axis=0) 

                training_true = np.append(training_true, trueExample, axis=0) 

                newTrainingLabel = np.expand_dims(training_true_labels[i], axis=0) 

                training_true_labels = np.append(training_true_labels, newTrainingLabel, axis=0) 
                                
                newTrainingIndex = trueLabels[i]
                training_set_index = np.append(training_set_index, newTrainingIndex, axis=0) 

                i += 1

            training_set  = np.append(training_false, training_true, axis=0) 
        
            training_set_labels = np.append(training_false_labels, training_true_labels, axis=0)
            
            training_set = np.squeeze(training_set, axis=1) 

            training_set_labels = np.squeeze(training_set_labels, axis=1)

            print("BALANCED training set", training_set.shape[0]) 
            print("BALANCED training set labels", training_set_labels.shape[0]) 
        
        shuffler = np.random.permutation(training_set.shape[0])
        training_set = training_set[shuffler]
        training_set_labels = training_set_labels[shuffler]
        training_set_index = training_set_index[shuffler]
        print(training_set.shape) 
        print(training_set_labels.shape)
        for _ in range(number_of_epochs):
                    
            outputMatrix, trainingOutput = medicalClassifier.forwardPropagate(training_set, training_set_labels)
                        
            medicalClassifier.backwardPropagate() 
 
        outputTrainingMatrix, trainingOutput = medicalClassifier.forwardPropagate(balancedDataset[0:4000], balancedDatasetLabels[0:4000])
 
        # outputValidationMatrix, validationOutput = medicalClassifier.forwardPropagate(miniValidationSet, miniValidationLabels)
 
        logLoss_trainingList.append(trainingOutput) 
 
        trainLosses.append(trainingOutput)
 
        nextImage = np.random.choice(balancedDatasetLabels[0:4000].shape[0], 25, replace=False)
 
        max_arr_training = outputTrainingMatrix 
        
        max_arr_training = np.argmax(max_arr_training, axis=1)
 
        standardMiniTrainingSet = np.argmax( balancedDatasetLabels[0:4000], axis=1)
 
        trainingCorrect = np.sum(np.equal(max_arr_training, standardMiniTrainingSet))
 
        trainingTotal = balancedDatasetLabels[0:4000].shape[0]
 
        trainingAccuracy = trainingCorrect / trainingTotal 
 
        print("training accuracy", trainingAccuracy)
        
        for i in range(max_arr_training.shape[0]): 
            if max_arr_training[i] ==   standardMiniTrainingSet[i] and max_arr_training[i] == 0: 
                 train_trueNegative += 1 
            elif max_arr_training[i] == standardMiniTrainingSet[i] and max_arr_training[i] == 1: 
                 train_truePositive += 1 
            elif max_arr_training[i] != standardMiniTrainingSet[i] and max_arr_training[i] == 0: 
                train_falseNegative += 1 
            else: 
                train_falsePositive += 1 
        
        if train_falsePositive == 0: 
            train_falsePositive = 1 

        if train_trueNegative == 0: 
            train_trueNegative = 1 

        if train_truePositive == 0: 
            train_truePositive = 1

        if train_falseNegative == 0: 
            train_falseNegative = 1
        specificity_train = train_trueNegative / (train_trueNegative + train_falsePositive)
        sensitivity_train = train_truePositive / (train_truePositive + train_falseNegative)
        
        specificityAccuracy.append(specificity_train) 
        sensitivityAccuracy.append(sensitivity_train) 
        
        print("SPECIFICITY ACCURACY", specificity_train) 
        print("SENSITIVITY ACCURACY", sensitivity_train)

        # max_arr_validation = np.argmax(max_arr_validation, axis=1)
 
        # standardMiniValidation = np.argmax(miniValidationLabels, axis=1)
 
 
        # validationCorrect = np.sum(np.equal(max_arr_validation,standardMiniValidation))
 
        # validationTotal = miniValidationLabels.shape[0]
 
        # validationAccuracy = validationCorrect / validationTotal 
 
        # print("validation accuracy", validationAccuracy)
 
        trainAccuracies.append(trainingAccuracy)
 
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(trainLosses,label="training loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("TrainingValidationLoss_RegularTraining.png")
 
    plt.figure(figsize=(10,5))
    plt.title("Training and Testing Accuracy")
    plt.plot(trainAccuracies,label="Training Accuracy")
    plt.xlabel("iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    plt.savefig("TrainingTestingAccuracy_RegularTraining.png")
 
    plt.figure(figsize=(10,5)) 
    plt.title("Specificity and Sensitivity Accuracy") 
    plt.plot(specificityAccuracy, label="Specificity Accuracy %") 
    plt.plot(sensitivityAccuracy, label="Sensitivity Accuracy %") 
    plt.legend() 
    plt.show() 
    plt.savefig("SpecificitySensitivityACcuracy_RegularTraining.png")
        
 
            
 
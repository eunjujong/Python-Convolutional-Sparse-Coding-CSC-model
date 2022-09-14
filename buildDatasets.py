import numpy as np 
import pandas as pd 
from PIL import Image 
import os

# Citation: YXD/dsalaj on stackoverflow
# https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
# Convert Array of Indices To 1 Hot Encoded Numpy Array
def one_hot(a): 
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size), a] = 1 
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
 
    validation_set = None
 
    validationSet_labels = []

    for i in range(validationSetNames.shape[0]): 
 
        abspath = os.path.abspath("./TrainingSet128/" + validationSetNames[i] + ".jpg")
 
        trainingInstance = Image.open(abspath)
 
        np_image = np.array(trainingInstance)
 
        trainingInstance = np.reshape(trainingInstance, (1, 32, 32)) 
 
        print("VALIDATION INSTANCE CREATED, ITERATION #{}".format(i))
 
        trainingInstance =  np.expand_dims(trainingInstance, axis=0)
 
        if i == 0:
 
            validation_set = trainingInstance
 
            newLabelCheck = validationSetLabels[i]
 
            if len(newLabelCheck) == 2: 
                validationSet_labels.append(0)
            else: 
                validationSet_labels.append(1)
 
        else: 
 
            validation_set = np.append(validation_set, trainingInstance, axis=0) 
 
            newLabelCheck = validationSetLabels[i]
 
            if len(newLabelCheck) == 2: 
                validationSet_labels.append(0)
            else: 
                validationSet_labels.append(1)
 
    dataset = None
 
    dataset_labels = []
 
    for i in range(trainingSetNames.shape[0]): 
 
        abspath = os.path.abspath("./TrainingSet128/" + trainingSetNames[i] + ".jpg")
        try: 
            trainingInstance = Image.open(abspath)
        except FileNotFoundError: 
            continue 
 
        np_image = np.array(trainingInstance)
 
        trainingInstance = np.reshape(trainingInstance, (1, 32, 32)) 
 
        print("TRAINING INSTANCE CREATED, ITERATION #{}".format(i))
 
        trainingInstance =  np.expand_dims(trainingInstance, axis=0)
 
        if i == 0:
 
            dataset = trainingInstance
 
            newLabelCheck = trainingSetLabels[i]
 
            if len(newLabelCheck) == 2: 
                dataset_labels.append(0)
            else: 
                dataset_labels.append(1)
 
        else: 
 
            dataset = np.append(dataset, trainingInstance, axis=0) 
 
            newLabelCheck = trainingSetLabels[i]
 
            if len(newLabelCheck) == 2: 
                dataset_labels.append(0)
            else: 
                dataset_labels.append(1)

    preservedDatasetLabels  = np.array(dataset_labels)
        
    trueLabels = np.argwhere(preservedDatasetLabels == 1) 

    nonTrueLabels = np.argwhere(preservedDatasetLabels == 0) 

    nonTrueDataset = dataset[nonTrueLabels] 

    trueDataset = dataset[trueLabels]
    print(trueLabels) 
    print(nonTrueLabels)
    print("NON TRUE DATASET", preservedDatasetLabels[nonTrueLabels]) 
    print("TRUE DATASET", preservedDatasetLabels[trueLabels])

    i = 0
    
    trueDatasetLabels = preservedDatasetLabels[trueLabels]

    nonTrueDatasetLabels = preservedDatasetLabels[nonTrueLabels]
    
    while nonTrueDataset.shape[0] != trueDataset.shape[0]: 
        
        trueExample = np.expand_dims(trueDataset[i], axis=0)

        trueDataset = np.append(trueDataset, trueExample, axis=0)
        
        newTrueLabel = np.expand_dims(trueDatasetLabels[i], axis=0) 

        trueDatasetLabels = np.append(trueDatasetLabels, newTrueLabel, axis=0)
        
        i += 1

    print("FALSE DATASET", nonTrueDataset.shape[0]) 
    print("TRUE DATASET", trueDataset.shape[0])

    balancedDataset = np.append(trueDataset, nonTrueDataset, axis=0) 
    
    balancedDatasetLabels = np.append(trueDatasetLabels, nonTrueDatasetLabels, axis=0)
    print(balancedDatasetLabels)
    balancedDatasetLabels = np.squeeze(balancedDatasetLabels, axis=1)
    balancedDatasetLabelsIndex = balancedDatasetLabels 
    print(balancedDatasetLabels.shape)
    balancedDatasetLabels = one_hot(balancedDatasetLabelsIndex) 
    print("FULL BALANCED DATASET LABELS", balancedDatasetLabels)
    print("FULL BALANCED DATASET", balancedDataset.shape)
    print("FULL BALANCED DATASET LABELS", balancedDatasetLabels.shape)
    print("FULL BALANCED balancedDataset index", balancedDatasetLabelsIndex.shape)
    balancedDataset = np.squeeze(balancedDataset, axis=1) 
    with open('balancedDataset.npy', 'wb') as f: 
            np.save(f, balancedDataset) 

    with open('balancedDatasetLabels.npy', 'wb') as f: 
           np.save(f, balancedDatasetLabels) 
    
    with open('validation_set.npy', 'wb') as f: 
           np.save(f, validation_set) 

    with open('validationSet_labels.npy', 'wb') as f: 
           np.save(f, validationSet_labels)
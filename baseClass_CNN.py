import numpy as np 
import math 

class ConvolutionLayer: 

    def __init__(self, num_filters, kernel_size, stride=1, padding=0, batch_size=1, channels=1): 
        self.num_filters = num_filters 
        self.kernel_size = kernel_size 
        self.stride = stride 
        self.padding = padding 
        self.batch_size = batch_size 
        self.channels = channels
        self.lr = 1e-3

        inputWidth = 64
        inputHeight = 64
        self.outputWidth = math.floor((inputWidth - self.kernel_size[0] + 2 * self.padding) / self.stride) + 1 
        self.outputHeight = math.floor((inputHeight - self.kernel_size[1] + 2 * self.padding) / self.stride) + 1

        self.weight = np.random.random((self.kernel_size[0], self.kernel_size[1], self.channels, self.num_filters)) * 0.01
        self.bias = np.random.random((1, 1, 1, self.num_filters)) * 0.01
        # https://stackoverflow.com/questions/62249084/what-is-the-numpy-equivalent-of-tensorflow-xavier-initializer-for-cnn
        # Applies Xavier Initialization 

    def forwardPropagate(self, X): 
        self.inputData = X
        if self.padding > 0:
            self.inputData = np.pad(self.inputData , ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant', constant_values=(0,0))

        featureMaps = np.zeros((X.shape[0], self.outputWidth, self.outputHeight, self.num_filters))

        numberBatches = X.shape[0]
        numberOutputWidth = self.outputWidth 
        numberOutputHeight = self.outputHeight 
        filterWidth = self.kernel_size[0]
        filterHeight = self.kernel_size[1] 

        for b in range(numberBatches): 
            for w in range(numberOutputWidth): 
                filterStart_width= w * self.stride 
                filterEnd_width = filterStart_width + filterWidth 
                for h in range(numberOutputHeight): 
                    filterStart_height = w * self.stride 
                    filterEnd_height = filterStart_height + filterHeight 
                    for filter in range(self.num_filters): 
                        featureMaps[b, w, h, filter] = np.sum(self.inputData[b,filterStart_width:filterEnd_width, filterStart_height:filterEnd_height, :]*self.weight[:,:,:,filter] + self.bias[:, :, :, filter])
        return featureMaps 
    
    def backwardsPropagate(self, gradient):
        self.inputWidth = self.inputData.shape[1]
        self.inputHeight = self.inputData.shape[2]

        gradientFeatureMaps = np.zeros((self.inputData.shape[0], self.inputWidth, self.inputHeight, self.num_filters))

        numberBatches = self.inputData.shape[0]
        numberOutputWidth = self.outputWidth 
        numberOutputHeight = self.outputHeight 
        filterWidth = self.kernel_size[0]
        filterHeight = self.kernel_size[1]

        self.dW = np.zeros(self.weight.shape)
        self.db = np.zeros(self.bias.shape)
        for b in range(numberBatches): 
            for w in range(numberOutputWidth): 
                filterStart_width= w * self.stride 
                filterEnd_width = filterStart_width + filterWidth 
                for h in range(numberOutputHeight): 
                    filterStart_height = w * self.stride 
                    filterEnd_height = filterStart_height + filterHeight 


                    for filter in range(self.num_filters): 
                        gradientFeatureMaps[b, filterStart_width:filterEnd_width, filterStart_height:filterEnd_height, :] += np.multiply(self.weight[:, :, :, filter], gradient[b, w, h, filter])
                        self.dW[:, :, :, filter] += np.multiply(self.inputData[b, filterStart_width:filterEnd_width, filterStart_height:filterEnd_height, :], gradient[b, w, h, filter])    
                        self.db[:, :, :, filter] += gradient[b, h, w, filter]
        self.weight -= self.dW * self.lr 
        self.bias -= self.db * self.lr
        if self.padding > 0: 
            gradientFeatureMaps = gradientFeatureMaps[:, self.padding:-self.padding, self.padding:-self.padding, :]
        return gradientFeatureMaps 
     


class MaxPoolingLayer: 

    def __init__(self, num_filters, kernel_size, stride=1, padding=0): 
        self.num_filters = num_filters 
        self.kernel_size = kernel_size 
        self.stride = stride 
        self.padding = padding 

    def maxPool(self, X):
        return np.max(X)  

    def forwardPropagate(self, X):
        inputWidth = X.shape[1]
        inputHeight = X.shape[2] 

        self.outputWidth = math.floor((inputWidth - self.kernel_size[0]) / self.stride) + 1 
        self.outputHeight = math.floor((inputHeight - self.kernel_size[1]) / self.stride) + 1

        maxPoolFeatureMaps = np.zeros((X.shape[0], self.outputWidth, self.outputHeight, self.num_filters))

        numberBatches = X.shape[0]
        numberOutputWidth = self.outputWidth 
        numberOutputHeight = self.outputHeight 
        filterWidth = self.kernel_size[0]
        filterHeight = self.kernel_size[1] 

        self.inputData = X 

        for b in range(numberBatches): 
            for w in range(numberOutputWidth): 
                filterStart_width= w * self.stride 
                filterEnd_width = filterStart_width + filterWidth 
                for h in range(numberOutputHeight): 
                    filterStart_height = w * self.stride 
                    filterEnd_height = filterStart_height + filterHeight 

                    for filter in range(self.num_filters): 
                        maxPoolFeatureMaps[b, w, h, filter] = self.maxPool(X[b, filterStart_width:filterEnd_width, filterStart_height:filterEnd_height, filter])
        return maxPoolFeatureMaps 

    def backwardsPropagate(self, gradient):
        self.inputWidth = self.inputData.shape[1]
        self.inputHeight = self.inputData.shape[2]
        gradientFeatureMaps = np.zeros((self.inputData.shape[0], self.inputWidth, self.inputHeight, self.num_filters))
        

        numberBatches = self.inputData.shape[0]
        numberOutputWidth = self.outputWidth 
        numberOutputHeight = self.outputHeight 
        filterWidth = self.kernel_size[0]
        filterHeight = self.kernel_size[1] 

        for b in range(numberBatches): 
            for w in range(numberOutputWidth): 
                filterStart_width= w * self.stride 
                filterEnd_width = filterStart_width + filterWidth 
                for h in range(numberOutputHeight): 
                    filterStart_height = w * self.stride 
                    filterEnd_height = filterStart_height + filterHeight 

                    for filter in range(self.num_filters): 
                        window = self.inputData[b, filterStart_width:filterEnd_width, filterStart_height:filterEnd_height, filter]
                        mask = (window == np.max(window))
                        gradientFeatureMaps[b, filterStart_width:filterEnd_width, filterStart_height:filterEnd_height, filter] += np.multiply(mask, gradient[b, w, h, filter])

        return gradientFeatureMaps 

class MLP: 

    def __init__(self, numberofFCLayers, numberOfHiddenLayers, sizes, lr, lambdaValue): 
        self.listOfLayers = []
        #self.inputLayer = InputLayer(meanX, stdX) 
        self.fcLayers = []
        self.activationLayers = []
        self.lr = lr
        self.lambdaValue = lambdaValue

        for i in range(numberofFCLayers-1):
            self.fcLayers.append(FCLayer(sizes[i], sizes[i+1], self.lr, self.lambdaValue))
            self.activationLayers.append(ReLuLayer()) 
            self.listOfLayers.append(self.fcLayers[i])
            self.listOfLayers.append(self.activationLayers[i])
        self.fcLayers.append(FCLayer(sizes[-2], sizes[-1], self.lr, self.lambdaValue))
        self.listOfLayers.append(self.fcLayers[-1])
        self.listOfLayers.append(SigmoidLayer())
        self.logLoss = None 
        self.sizes = sizes
        self.weightSubstitute = self.fcLayers[0].getWeights() 
        self.numberOfFCLayers = len(self.fcLayers)
        self.numberOfLayers = len(self.listOfLayers)
    
    def forwardPropagate(self, X, Y): 
        outputFCLayer = self.fcLayers[0].forwardPropagate(X) 

        output = self.activationLayers[0].forwardPropagate(outputFCLayer) 

        newFC_layer = None 

        newOutput = output
        
        for i in range(1, self.numberOfFCLayers-1): 
            temp = newOutput
            newFC_layer = self.fcLayers[i].forwardPropagate(temp) 
            newOutput = self.activationLayers[i].forwardPropagate(newFC_layer) 
            temp = newOutput

        newOutput = self.fcLayers[-1].forwardPropagate(newOutput) 

        newOutput = self.listOfLayers[-1].forwardPropagate(newOutput)

        self.logLoss = LogLoss(Y, self.weightSubstitute, 0) 
        self.logLoss.forwardPropagate(newOutput)

        output = self.logLoss.eval()

        return newOutput, output


    def backwardPropagate(self): 
        gradientObj = self.logLoss.gradient() 

        m = self.numberOfLayers

        listOfLayers = self.listOfLayers

        for i in range(m - 1, -1, -1):
                gradientObj = listOfLayers[i].backwardsPropagate(gradientObj)
        return gradientObj

    def getFCLayers(self): 
        return self.fcLayers

class FlattenLayer: 

    def __init__(self): 
        self.inputData = None

    def forwardPropagate(self, X): 
        self.inputData = X
        self.savedState = self.inputData.shape
        return np.reshape(X, (self.inputData.shape[0], int(self.inputData.size / self.inputData.shape[0])))
    
    def backwardsPropagate(self, gradient): 
        return np.reshape(gradient, self.savedState)

class InputLayer:

	def __init__(self, meanX, stdX):
		self.meanX = meanX
		self.stdX = stdX


	def forwardPropagate(self, X):
		norm_X = X / 255.0
		return norm_X


class ReLuLayer:
    def __init__(self): 
        self.inputData = None

    def ReLuFunc(self, X):
        maximumX = X * (X > 0)
        return maximumX

    def forwardPropagate(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        self.inputData = X

        output = self.ReLuFunc(X)

        return output
    
    def gradient(self): 
        gradient = np.copy(self.inputData) 

        return 1. * (gradient > 0)

    def backwardsPropagate(self, gradient): 
        
        return np.multiply(self.gradient(), gradient)


class SigmoidLayer:
    def __init__(self): 
        self.inputData = None
        self.flag = "SIGMOID"

    def SigmoidFunc(self, X):
        return 1 / (1 + np.exp(-X))

    def forwardPropagate(self, X):        
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        self.inputData = X
        output = self.SigmoidFunc(X)

        return output
    
    def gradient(self): 
        output = self.forwardPropagate(self.inputData) 
        return np.multiply(output, (1-output))

    def backwardsPropagate(self, gradient): 
        return np.multiply(self.gradient(), gradient)

class SoftmaxLayer:
    def __init__(self): 
        self.inputData = None

    def SoftMaxFunc(self, X):

        softmaxValue = np.zeros(X.shape) 

        for i in range(X.shape[0]): 
            softmaxValue[i,:] = np.exp(X[i,:]) / np.sum(np.exp(X[i,:]), axis=0)
        return softmaxValue 

    def forwardPropagate(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if len(X.shape) == 1: 
            X = np.reshape(X, (X.shape[0], 1))
        
        if X.shape[0] > 1 and X.shape[1] == 1:
            X = X.transpose()

        self.inputData = X

        output = self.SoftMaxFunc(X)

        return output
    
    def gradient(self): 

        output = self.forwardPropagate(self.inputData) 

        return np.multiply(output, (1-output)) 

    def backwardsPropagate(self, gradient): 
        return np.multiply(self.gradient(), gradient)
    
class FCLayer:

    def __init__(self, sizein, sizeout, lr=0, lambdaValue=0):
        np.random.seed(0)
        self.weight = np.random.uniform(-0.001, 0.001, (sizein, sizeout))
        self.bias = np.random.uniform(-0.001, 0.001, (1, sizeout))
        self.sizein = sizein
        self.sizeout = sizeout
        self.inputData = None
        self.s = 0 
        self.r = 0 
        self.pho1 = 0.9
        self.pho2 = 0.999 
        self.sigma = 1e-7
        self.newGrad = None
        self.t = 0
        self.lr = lr
        self.lambdaValue = lambdaValue
        self.batch_size = 0
    
    def getWeights(self): 
        return self.weight

    def forwardPropagate(self, X):
        self.inputData = X
        self.batch_size = X.shape[0]
        outputMatrix = np.dot(X, self.weight) + self.bias
        return outputMatrix

    def gradient(self): 
        return self.weight.transpose()

    def backwardsPropagate(self, gradient):

        self.newGrad = gradient 

        gradient = np.dot(gradient,self.gradient())

        self.t += 1 

        gradientObj_weight = np.dot(self.inputData.transpose(), self.newGrad)
	
        self.s = self.s*self.pho1 + (1-self.pho1)*gradientObj_weight

        self.r = self.r*self.pho2 + (1-self.pho2)*(np.multiply(gradientObj_weight, gradientObj_weight))
       
        self.num = self.s / (1 - math.pow(self.pho1, self.t)) 

        self.denom = self.r / (1 - math.pow(self.pho2, self.t))  

        self.weight = self.weight - (self.lr*(((self.num) / (np.sqrt(self.denom) + self.sigma))) + self.lr*(self.lambdaValue)*self.weight)

        return gradient
    
class LeastSquares: 

    def __init__(self, target, weights, lambdaValue): 
        self.target = target
        self.input = None
        self.listOfWeights = weights 
        self.lambdaValue = lambdaValue

    def forwardPropagate(self, input): 
        self.input = input 

        return self.input 

    def eval(self): 
        output = (self.target - self.input)**2 / self.input.shape[0]
        J = output + (self.lambdaValue/2*self.input.shape[0])*(np.sum(np.square(self.listOfWeights))) 
        return J
    
    def gradient(self): 
        return 2*(self.input - self.target)

class LogLoss: 

    def __init__(self, target, weights, lambdaValue, l2="No"): 
        self.target = target
        self.input = None
        self.alpha = 0.5
        self.l2 = l2
        self.listOfWeights = np.array(weights) 
        self.lambdaValue = lambdaValue 
        self.booleanValue = (self.lambdaValue == 0)

    def forwardPropagate(self, input): 
        self.input = input

        return self.input 

    def eval(self): 
        output = -(self.target*np.log(self.input) + (1-self.target)*np.log(1-self.input))
        sumOutput = np.sum(output, axis=0) 
        if not self.booleanValue:
            J = sumOutput + (self.lambdaValue/2)*(np.sum(np.square(self.listOfWeights)))
        else: 
             J = np.sum(sumOutput,axis=0)
        print(np.sum(J, axis=0) / self.input.shape[0], "LOG LOSS FUNCTION")
        return np.sum(J, axis=0) / self.input.shape[0]

    def gradient(self):
        return ((1 - self.target) / (1 - self.input)) - (self.target / self.input)

class CrossEntropyLoss: 

    def __init__(self, target, weights, lambdaValue): 
        if not isinstance(target, np.ndarray): 
            target = np.array(target)
        self.target = target
        self.input = None
        self.listOfWeights = weights 
        self.lambdaValue = lambdaValue

    def forwardPropagate(self, input): 
        if not isinstance(input, np.ndarray): 
            input = np.array(input) 

        self.input = input 

        return self.input 

    def eval(self):       
        evalOutput = -self.target * np.log(self.input)
        total = np.sum(evalOutput, axis=1)
        total = np.sum(total, axis=0) / self.input.shape[0]
        J = total + (self.lambdaValue/2*self.input.shape[0])*(np.sum(np.square(self.listOfWeights))) 
        return J 

    def gradient(self): 
        return (-1*self.target / self.input)
import math, numpy as np, random

class ConvLayer:

  def __init__(self, num_filters, filter_width, stride = 1, padding = 0):
    self.fw = filter_width; self.n_f = num_filters; self.s = stride; self.p = padding
    self.W = None; self.b = None
    self.lr = 1e-3 
    self.lambdaValue = 5.12

  def forward_propagate(self, input):
    if self.p > 0:                                  # pad input
      shape = ((0, 0), (self.p, self.p), (self.p, self.p), (0, 0))
      input = np.pad(input, shape, mode='constant', constant_values = (0, 0))
    im, ih, iw, id = input.shape; s = self.s; fw = self.fw; f = self.n_f
    self.input_shape = input.shape
    if self.W is None:
      self.W = np.random.random((self.fw, self.fw, id, self.n_f)) * 0.001
      self.b = np.random.random((1, 1, 1, self.n_f)) * 0.0001
    self.n_rows = math.ceil(min(fw, ih - fw + 1) / s)
    self.n_cols = math.ceil(min(fw, iw - fw + 1) / s)
    z_h = int(((ih - fw) / s) + 1); z_w = int(((iw - fw) / s) + 1)
    self.Z = np.empty((im, z_h, z_w, f)); self.input_blocks = []

    for t in range(self.n_rows): 
      self.input_blocks.append([])
      b = ih - (ih - t) % fw
      cols = np.empty((im, int((b - t) / fw), z_w, f))
      for i in range(self.n_cols):
        l = i * s; r = iw - (iw - l) % fw
        block = input[:, t:b, l:r, :]
        block = np.array(np.split(block, (r - l) / fw, 2))
        block = np.array(np.split(block, (b - t) / fw, 2))
        block = np.moveaxis(block, 2, 0)
        block = np.expand_dims(block, 6)
        self.input_blocks[t].append(block)
        block = block * self.W
        block = np.sum(block, 5)
        block = np.sum(block, 4)
        block = np.sum(block, 3)
        cols[:, :, i::self.n_cols, :] = block
      self.Z[:, t * s ::self.n_rows, :, :] = cols
    self.Z += self.b
    return self.Z

  def backwardsPropagate(self, dZ):
    im, ih, iw, id = self.input_shape; s = self.s; fw = self.fw; f = self.n_f
    n_rows = self.n_rows; n_cols = self.n_cols
    dA_prev = np.zeros((im, ih, iw, id))
    dW = np.zeros(self.W.shape); db = np.zeros(self.b.shape)
    for t in range(n_rows):
      row = dZ[:, t::n_rows, :, :]
      for l in range(n_cols):
        b = (ih - t * s) % fw; r = (iw - l * s) % fw  # region of input and dZ for this block
        block = row[:, :, l * s::n_cols, :]           # block = corresponding region of dA
        block = np.expand_dims(block, 3)              # axis for channels
        block = np.expand_dims(block, 3)              # axis for rows
        block = np.expand_dims(block, 3)              # axis for columns
        dW_block = block * self.input_blocks[t][l]
        dW_block = np.sum(dW_block, 2)
        dW_block = np.sum(dW_block, 1)
        dW_block = np.sum(dW_block, 0)
        dW += dW_block
        db_block = np.sum(dW_block, 2, keepdims=True)
        db_block = np.sum(db_block, 1, keepdims=True)
        db_block = np.sum(db_block, 0, keepdims=True)
        db += db_block
        dA_prev_block = block * self.W
        dA_prev_block = np.sum(dA_prev_block, 6)
        dA_prev_block = np.reshape(dA_prev_block, (im, ih - b - t, iw - r - l, id))
        dA_prev[:, t:ih - b, l:iw - r, :] += dA_prev_block
    self.W -= dW*self.lr
    self.b -= (db * self.lr)
    if self.p > 0:                                   # remove padding
      dA_prev = dA_prev[:, self.p:-self.p, self.p:-self.p, :]
    return dA_prev



class PoolLayer:

  def __init__(self, filter_width, stride = 1):
    self.fw = filter_width; self.s = stride
  
  def forward_propagate(self, input):
    im, ih, iw, id = input.shape; fw = self.fw; s = self.s
    self.n_rows = math.ceil(min(fw, ih - fw + 1) / s)
    self.n_cols = math.ceil(min(fw, iw - fw + 1) / s)
    z_h = int(((ih - fw) / s) + 1); z_w = int(((iw - fw) / s) + 1)
    self.Z = np.empty((im, z_h, z_w, id)); self.input = input
    for t in range(self.n_rows): 
      b = ih - (ih - t) % fw
      Z_cols = np.empty((im, int((b - t) / fw), z_w, id))
      for i in range(self.n_cols):
        l = i * s; r = iw - (iw - l) % fw
        block = input[:, t:b, l:r, :]
        block = np.array(np.split(block, (r - l) / fw, 2))
        block = np.array(np.split(block, (b - t) / fw, 2))
        block = self.pool(block, 4)
        block = self.pool(block, 3)
        block = np.moveaxis(block, 0, 2)
        block = np.moveaxis(block, 0, 2)
        Z_cols[:, :, i::self.n_cols, :] = block
      self.Z[:, t * s ::self.n_rows, :, :] = Z_cols
    return self.Z

  def assemble_block(self, block, t, b, l, r):
    ih = self.input.shape[1]; iw = self.input.shape[2]
    block = np.repeat(block, self.fw ** 2, 2)
    block = np.array(np.split(block, block.shape[2] / self.fw, 2))
    block = np.moveaxis(block, 0, 2)
    block = np.array(np.split(block, block.shape[2] / self.fw, 2))
    block = np.moveaxis(block, 0, 3)
    return np.reshape(block, (self.input.shape[0], ih - t - b, iw - l - r, self.input.shape[3]))

class PoolLayer_Max(PoolLayer):

  def __init__(self, filter_width, stride = 1):
    self.pool = np.max
    super().__init__(filter_width, stride)
  
  def backwardsPropagate(self, dZ):
    im, ih, iw, id = self.input.shape
    fw = self.fw; s = self.s; n_rows = self.n_rows; n_cols = self.n_cols
    dA_prev = np.zeros(self.input.shape)
  
    for t in range(n_rows):
      mask_row = self.Z[:, t::n_rows, :, :]
      row = dZ[:, t::self.n_rows, :, :]
      for l in range(self.n_cols):
        b = (ih - t * s) % fw; r = (iw - l * s) % fw
        mask = mask_row[:, :, l * s::n_cols, :]
        mask = self.assemble_block(mask, t, b, l, r)
        block = row[:, :, l * s::n_cols, :]
        block = self.assemble_block(block, t, b, l, r)
        mask = (self.input[:, t:ih - b, l:iw - r, :] == mask)
        dA_prev[:, t:ih - b, l:iw - r, :] += block * mask
    return dA_prev

class PoolLayer_Avg(PoolLayer):

  def __init__(self, filter_width, stride = 1):
    self.pool = np.mean
    super().__init__(filter_width, stride)

  def backpropagate(self, dZ, learning_rate):
    im, ih, iw, id = self.input.shape
    fw = self.fw; s = self.s; n_rows = self.n_rows; n_cols = self.n_cols
    dA_prev = np.zeros(self.input.shape)
  
    for t in range(n_rows):
      row = dZ[:, t::n_rows, :, :]
      for l in range(n_cols):
        b = (ih - t * s) % fw; r = (iw - l * s) % fw
        block = row[:, :, l * s::n_cols, :]
        block = self.assemble_block(block, t, b, l, r)
        dA_prev[:, t:ih - b, l:iw - r, :] += block / (fw ** 2)
    return dA_prev

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
        self.epsilon = 1e-7

    def forwardPropagate(self, input): 
        self.input = input

        return self.input 

    def eval(self): 
        output = -(self.target*np.log(self.input + self.epsilon) + (1-self.target)*np.log(1-self.input + self.epsilon))
        sumOutput = np.sum(output, axis=0) 
        if not self.booleanValue:
            J = sumOutput + (self.lambdaValue/2)*(np.sum(np.square(self.listOfWeights)))
        else: 
             J = np.sum(sumOutput,axis=0)
        print(np.sum(J, axis=0) / self.input.shape[0], "LOG LOSS FUNCTION")
        return np.sum(J, axis=0) / self.input.shape[0]

    def gradient(self):
        return ((1 - self.target) / (1 - self.input + self.epsilon)) - (self.target / self.input + self.epsilon)

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
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:43:12 2020

@author: dell
"""


import numpy as np
import pandas as pd

class MLP(object):
    def __init__(self, NI, NH, NO):   #     initialise the attributes
        self.no_in = NI               #     NI number of input
        self.no_hidden = NH           #     NH number of Hiddlen units
        self.no_out = NO              #     NO number of output
        self.W1 = np.array        #     An matrix containing the weights in the lower layer
        self.W2 = np.array        #     An matrix containing the weights in the upper layer
        self.dW1 = np.array       #     An matrix containing the weights change to be applied on w1 in the lower layer
        self.dW2 = np.array       #     An matrix containing the weights change to be applied on w2 in the upper layer
        self.Z1 = np.array        #     An array containing the activations in the lower layer
        self.Z2 = np.array        #     An array containing the activations in the upper layer
        self.bias1 = np.array       #     Bias for the lower layer
        self.bias2 = np.array       #     Bias for the upper layer
        self.dBias1 = np.array    #     An matrix containing the bias change to be applied on w1 in the upper layer
        self.dBias2 = np.array    #     An matrix containing the bias change to be applied on w2 in the upper layer
        self.H = np.array  #  array where the values of the hidden neurons are stored – need these saved to compute dW2)
        self.O = np.array  #  array where the outputs are stored   
        
        
    #  Initialize matrix W1 and W2 randomly from Normal distribution having mean 0 and variance 1     
    def randomise(self): 
        self.W1 = np.array((np.random.uniform(low=0, high=1, size=(self.no_in, self.no_hidden))).tolist())
        self.W2 = np.array((np.random.uniform(low=0, high=1, size=(self.no_hidden, self.no_out))).tolist())
        

        # set dW1 and dW2 to all zeroes.
        self.dW1 = np.dot(self.W1, 0)
        self.dW2 = np.dot(self.W2, 0)
        
    # Define a logistic sigmoid function which takes input sigInput and returns 1/(1 + math.exp(-sigInput)).
    def sigmoid(self, sigInput):
        return 1 / (1 + np.exp(-sigInput))
    def derivative_sigmoid(self, sigInput):
        return np.exp(-sigInput) / (1 + np.exp(-sigInput)) ** 2
    
    # Define a logistic tanH function which takes input sigInput and returns 2 / (1 + np.exp(-2*tangInput))-1.
    def tanh(self, tangInput):
        return (2 / (1 + np.exp(tangInput * -2))) - 1
#        return (np.exp(tangInput)-np.exp(-tangInput))/(np.exp(tangInput)+np.exp(-tangInput))
    def derivative_tanH(self, tangInput):
        return 1 - (np.power(self.tanh(tangInput), 2))
        

        
      # Forward pass. Input vector I is processed to produce an output, which is stored in O[].
    def forward(self, I, activation):
    # If we use sigmoid activation function, take the inputs, and put them through the formula to get lower neuron's output
        if activation == 'sigmoid':
            # Array containing the activations in the lower layer
            self.Z1 = np.dot(I, self.W1)
            # Array where the values of the hidden neurons are stored 
            self.H = self.sigmoid(self.Z1)
            
            # Take lower layer's outputs, and put them through the formula to get upper neuron's output
            self.Z2 = np.dot(self.H, self.W2)
            # Array where the outputs are stored
            self.O = self.sigmoid(self.Z2)
    
        elif activation == 'tanh' :
            self.Z1 = np.dot(I, self.W1) 
            # Array where the values of the hidden neurons are stored 
            self.H = self.tanh(self.Z1)
            # If we use tanh activation function,, take lower layer's outputs, and put them through the formula to get upper neuron's output
            self.Z2 = np.dot(self.H, self.W2)
            # Array where the outputs are stored       
            self.O = self.Z2
#             print("tanhforward is" + self.O)
        return self.O
    
    
    #  backward pass 
    #  target is the output that we want, self.O is the output predicted by our network
    def backward(self, I, target, activation):
        output_error = np.subtract(target, self.O) #difference (error) in output 
        if activation == 'sigmoid' : 
            activation_O=self.derivative_sigmoid(self.Z2)
            activation_H=self.derivative_sigmoid(self.Z1)
        elif activation == 'tanh' :
            activation_O=self.derivative_tanH(self.Z2)
            activation_H=self.derivative_tanH(self.Z1)
        dw2_a = np.multiply(output_error, activation_O)
        self.dW2 = np.dot(self.H.T, dw2_a)
        dw1_a=np.multiply(np.dot(dw2_a, self.W2.T), activation_H)
        self.dW1=np.dot(I.T,dw1_a)
        return np.mean(np.abs(output_error))


    # Adjust the weights
    def updateWeights(self, learningRate):
        self.W1 = np.add(self.W1,learningRate * self.dW1)
        self.W2 = np.add(self.W2,learningRate * self.dW2)
        self.dW1 = np.array
        self.dW2 = np.array



# =============================================================================
# XOR Test
# =============================================================================

log = open("xortest.txt", "w")
print("XOR TEST\n", file = log)


def XOR(max_epochs, learning_rate):
    np.random.seed(1)
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([[0], [1], [1], [0]])

    NI = 2
    NH = 4
    NO = 1
    NN = MLP(NI, NH, NO)

    NN.randomise()
    print('\nMax Epoch:\n' + str(max_epochs), file=log)
    print('\nLearning Rate:\n' + str(learning_rate), file=log)
    print('\nBefore Training:\n', file=log)
    for i in range(len(inputs)):
        NN.forward(inputs[i],'sigmoid')
        print('Target:\t {}  Output:\t {}'.format(str(outputs[i]), str(NN.O)), file=log)
    print('\nTraining:\n', file=log)
    
    for i in range(0, max_epochs):
        NN.forward(inputs,'sigmoid')
        error = NN.backward(inputs, outputs,'sigmoid')
        NN.updateWeights(learning_rate)

        if (i + 1) % (max_epochs / 20) == 0:
            print(' Error at Epoch:\t' + str(i + 1) + '\t\t  is \t\t' + str(error), file=log)

    print('\n After Training :\n', file=log)
    
    accuracy=float(0)
    for i in range(len(inputs)):
        NN.forward(inputs[i],'sigmoid')
        print('Target:\t {}  Output:\t {}'.format(str(outputs[i]), str(NN.O)), file=log)
        if(outputs[i][0]==0):
            accuracy+=1-NN.O[0]
        elif(outputs[i][0]==1):
            accuracy+=NN.O[0]
    print('\nAccuracy:{}'.format(accuracy/4),file=log)
iteration=[10000,1000]
learn_rate=[1.0,0.8,0.6,0.4,0.2,0.02]

for i in range(len(iteration)):
    for j in range(len(learn_rate)):
        print('----------------------------------------------------------------------\n', file=log)
        XOR(iteration[i],learn_rate[j])
        print('\n-------------------------------------------------------------------\n', file=log)
        

        

# =============================================================================
# Generate 200 vectors containing 4 components each. The value of each 
# component should be a random number between -1 and 1. These will be 
# your input vectors. The corresponding output for each vector should be 
# the sin() of a combination of the components. Specifically, for inputs:
# [x1 x2 x3 x4]
# the (single component) output should be:
# sin(x1-x2+x3-x4)
# Now train an MLP with 4 inputs, at least 5 hidden units and one output
# on 150 of these examples and keep the remaining 50 for testing.
# =============================================================================

log = open("sintest.txt", "w")
print("SINE TEST\n", file = log)

def SIN(max_epochs, learning_rate, no_hidden):
    np.random.seed(213)
    inputs = []
    outputs = []
    for i in range(0, 200):
        four_inputs_vector = list(np.random.uniform(-1.0, 1.0, 4))
        four_inputs_vector = [float(four_inputs_vector[0]),float(four_inputs_vector[1]), 
                              float(four_inputs_vector[2]),float(four_inputs_vector[3])]
        inputs.append(four_inputs_vector)
    inputs=np.array(inputs)

    for i in range(200):
        outputs.append(np.sin([inputs[i][0] - inputs[i][1] + inputs[i][2] - inputs[i][3]]))

    no_in = 4
    no_out = 1
    NN = MLP(no_in, no_hidden, no_out)
    NN.randomise()
    print('\nMax Epoch:\n' + str(max_epochs), file=log)
    print('\nLearning Rate:\n' + str(learning_rate), file=log)
    print('\nBefore Training:\n', file=log)

    for i in range(150):
        NN.forward(inputs[i],'tanh')
        print('Target:\t{}\t Output:\t {}'.format(str(outputs[i]),str(NN.O)), file=log)
    print('Training:\n', file=log)
    
    
#    training process
    for i in range(0, max_epochs):
        error = 0
        NN.forward(inputs[:150],'tanh')
        error = NN.backward(inputs[:150], outputs[:150],'tanh')
        NN.updateWeights(learning_rate)
       #prints error every 5% of epochs
        if (i + 1) % (max_epochs / 20) == 0:
            print(' Error at Epoch:\t' + str(i + 1) + '\t  is \t' + str(error), file=log)
    
    difference=float(0)
    print('\n Testing :\n', file=log)
    for i in range(150, len(inputs)):
        NN.forward(inputs[i], 'tanh')
        print('Target:\t{}\t Output:\t {}'.format(str(outputs[i]), str(NN.O)), file=log)
        difference+=np.abs(outputs[i][0]-NN.O[0])

    accuracy=1-(difference/50)
    accuracylist.append(accuracy)
    print('\nAccuracy:{}'.format(accuracy),file=log)
    print('\ntestError:{}'.format(difference/50),file=log)
    

iteration=[100000]
learn_rate=[0.02,0.001,0.0006,0.0001]
accuracylist=[]
for i in range(len(iteration)):
    for j in range(len(learn_rate)):
         print('----------------------------------------------------------------------\n', file=log)
         SIN(iteration[i], learn_rate[j], no_hidden=10)
         print('\n-------------------------------------------------------------------\n', file=log)
print('Accuracylist:{}'.format(accuracylist),file=log)


# =============================================================================
# letter recognition
# =============================================================================



def letter(max_epochs, learning_rate):
    np.random.seed(1)
 
    inputs = []
    outputs = []
    doutput = []
    columns=["letter","x-box","y-box","width","height","onpix","x-bar","y-bar","x2bar","y2bar","xybar","x2ybr","xy2br","x-ege","xegvy","y-ege","yegvx"]
    
    df=pd.read_csv("letter-recognition.data", names=columns)
    doutput=df["letter"]
    
    
    for i in range(len(doutput)):
        outputs.append(ord(str(doutput[i]))-ord('A'))
    
    inputs=df.drop(["letter"], axis=1)
    inputs=np.array(inputs)
    inputs=inputs/15  #normalization
    
    #train set
    inputs_train=inputs[:16000]
    categorical_y = np.zeros((16000, 26))
    for i, l in enumerate(outputs[:16000]):
        categorical_y[i][l] = 1
    outputs_train=categorical_y
    
    #test set
    inputs_test=inputs[16000:]
#    categorical_y = np.zeros((4000, 26))
#    for i, l in enumerate(outputs[16000:]):
#        categorical_y[i][l] = 1
#    outputs_test=categorical_y
    
    #training process
    no_in= 16
    no_hidden = 10
    no_out = 26
    
    NN = MLP(no_in, no_hidden, no_out)
    NN.randomise()
    print('\nMax Epoch:\n' + str(max_epochs), file=log)
    print('\nLearning Rate:\n' + str(learning_rate), file=log)
    print('\nTraining Process:\n', file=log)
    
    for i in range(0, max_epochs):
        NN.forward(inputs_train,'tanh')
        error = NN.backward(inputs_train, outputs_train,'tanh')
        NN.updateWeights(learning_rate)
    
        if (i + 1) % (max_epochs / 20) == 0:
            print(' Error at Epoch:\t' + str(i + 1) + '\t  is \t' + str(error))
    
    
    #testing process
    def to_character0(outputvector):
        listov=list(outputvector)
        a=listov.index(max(listov))
        return chr(a+ord('A'))
    
    prediction=[]
    for i in range(4000):
        NN.forward(inputs_test[i],'tanh')
    #    print('Target:\t{}\t Output:\t{}'.format(str(outputs_test[i]),str(NN.O)))
    #    print('Target:\t{}\t Output:\t{}'.format(str(doutput[16000+i]),str(to_character0(NN.O))))
        prediction.append(to_character0(NN.O))
    
    
    
    def to_character(n):
        return chr(int(n) + ord('A'))
    
    correct = {to_character(i): 0 for i in range(26)}
    letter_num = {to_character(i): 0 for i in range(26)}
    
    print('==' * 30,file=log)
    for i, _ in enumerate(doutput[16000:]):
        letter_num[doutput[16000+i]] += 1
        # Print some predictions
        if i % 300 == 0:
            print('Expected: {} | Output: {}'.format(doutput[16000+i], prediction[i]),file=log)
        if doutput[16000+i] == prediction[i]:
            correct[prediction[i]] += 1
    
    print('==' * 30,file=log)
    # Calculate the accuracy
    accuracy = sum(correct.values()) / len(prediction)
    print('Test sample size: {} | Correctly predicted sample size: {}'.format(len(prediction),sum(correct.values())),file=log) 
    print('Accuracy: %.3f' % accuracy,file=log)
    
    # Performance on each class
    print('==' * 30,file=log)
    for k,v in letter_num.items():
        print('{} => Sample Number: {} | Correct Number: {} | Accuracy: {}'.format(k, v, correct[k], correct[k]/v),file=log)


iteration=[100000]
learn_rate=[0.000005]
for i in range(len(iteration)):
    for j in range(len(learn_rate)):
         print('----------------------------------------------------------------------\n', file=log)
         letter(iteration[i], learn_rate[j])
         print('\n-------------------------------------------------------------------\n', file=log)


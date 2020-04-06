# Вход 	 	Выход
# 0  0 	1 	  0
# 1  1 	1 	  1
# 1  0 	1 	  1
# 0  1 	1 	  0
import numpy as np
#activation function
def sigmoid(x,derivative = False):
    if(derivative == True):
        return np.exp(-x)/(1/np.exp(-x))^2
    return 1/(1+np.exp(-x))
#input data
start_input_data = np.array ([  [0,0,1],
                        [0,1,1],
                        [1,0,1],
                        [1,1,1] ])
#ouput data
start_output_data = np.array([[0,0,1,1]]).T
print("Training data\nInput:")
print(start_input_data)
print("Output:")
print(start_output_data)
np.random.seed(1)
#random weights for sinapses
synaptic_weights = 2 *np.random.random((3,1))-1
print("Random weights: ")
print(synaptic_weights)
#backpropagation method
for i in range(20000):
    input_layer = start_input_data
    output_data = sigmoid(np.dot(input_layer,synaptic_weights))
    err = start_output_data - output_data
    adj = np.dot(input_layer.T,err*(output_data*(1-output_data)))
    synaptic_weights += adj
#output after training
print("Result: ")
print(output_data)
#try give smthing New
new_input = np.array([1,1,0])
output_data = sigmoid(np.dot(new_input,synaptic_weights))
print("New data, which is not in training data:")
print(new_input)
print("Resut:")
print(output_data)
while(True):
    x = int(input("Write num from set {0,1}:"))
    y = int(input("Write one more num from set {0,1}:"))
    z = int(input("Write last num from set {0,1}:"))
    new_input = np.array([x,y,z])
    output_data = sigmoid(np.dot(new_input,synaptic_weights))
    print("New data, which is not in training data:")
    print(new_input)
    print("Resut:")
    print(output_data)

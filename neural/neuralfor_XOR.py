import numpy as np

# activation funtion hai relu bhe use kar skte hai
def segmoid(x):
    return 1/(1+np.exp(-x))

def segmoid_derivative(x):
    return x * ( 1 - x )

# XOR input and output
X = np.array([[0, 0, 0],
              [0, 1, 0],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [0],
              [0]])

np.random.seed(42)

input_layer = 3
hidden_layer = 2
output_layer = 1

w1 = np.random.rand(input_layer,hidden_layer)
w2 = np.random.rand(hidden_layer, output_layer)

b1 = np.random.rand(1,hidden_layer)
b2 = np.random.rand(1,output_layer)

learning_rate = 0.5
epochs = 1000

for epoch in range(epochs):

    z1 = np.dot(X,w1) + b1
    a1 = segmoid(z1)

    z2 = np.dot(a1,w2) + b2
    a2 = segmoid(z2)

    loss = np.mean((y-a2)**2)

# Backpropagation
    error = y - a2 # we did not take w2 because activation function (a2) of w2 will give more priceis ans seens the value are in 0 and 1
    d_a2 = error * segmoid_derivative(a2)

    hidden_error = d_a2.dot(w2.T) # we take w2 because w2 is weight from hidden layer to output layer and w2.T means transpose of w2
    d_a1 = hidden_error * segmoid_derivative(a1)

#Gradient Descent
    w2 += a1.T.dot(d_a2) * learning_rate
    b2 += np.sum(d_a2 , axis= 0 ) * learning_rate

    w1 += X.T.dot(d_a1) * learning_rate
    b1 += np.sum(d_a1 , axis= 0) * learning_rate

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

print("\nFinal predictions:")
print(a2.round())  # rounded output (0 or 1)

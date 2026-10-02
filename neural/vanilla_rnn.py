import numpy as np
from scipy.stats import probplot

from neural.neuralfor_XOR import hidden_layer


class RNN:
    def __init__(self,input_size,hidden_size,output_size,learning_rate=0.01):
        # initializing small weights random generated for every layer
        limit = np.sqrt(1.0 / hidden_size)
        self.Wi_h = np.random.randn(hidden_size,input_size)*0.01 # for small weight input -> hidden
        self.Wh_h = np.random.randn(hidden_size,hidden_size)*0.01 # hidden ->hidden
        self.Wh_o = np.random.randn(output_size,hidden_size)*0.01 # hidden -> output
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        # initializing bias for hidden layer and output layer
        self.b_h = np.zeros((hidden_size,1))
        self.b_o = np.zeros((output_size,1))

    # def forward(self,inputs):#inputs provided by the user
    #
    #     h = np.zeros((self.hidden_size,1))
    #     self.last_inputs = inputs# we are storing the input provided by the user in it so that we can access those input when ever we want
    #     self.last_hs = {-1  : h}#stores the sequence of hidden states across time (including the initial h_{-1}).
    #
    #     for t,x in enumerate(inputs):
    #         h = np.tanh(self.Wi_h @ x + self.Wh_h @ h + self.b_h)
    #         self.last_hs[t] = h # using t as index as enumerate generate index and output side by side
    #
    #     y = self.Wh_o @ h + self.b_o
    #     pred = self.softmax(y)
    #     return pred
    #
    # # softmax function to convert y->output from any real number to 0-1 fro probability
    # def softmax(self,y):
    #     exp_x = np.exp(y-np.max(y))
    #     return exp_x / np.sum(exp_x)

    def forward(self, inputs):  # inputs = list of input vectors from the user
        h = np.zeros((self.hidden_size, 1))  # initial hidden state (all zeros)

        # Save inputs for backprop
        self.last_inputs = inputs

        # Dictionary to store hidden states over time (start with h_{-1} = 0 vector)
        self.last_hs = {-1: h}

        # Forward pass through sequence
        for t, x in enumerate(inputs):
            # hidden state update:
            # Wi_h -> input to hidden weights
            # Wh_h -> hidden to hidden weights
            # @ = matrix multiplication
            h = np.tanh(self.Wi_h @ x + self.Wh_h @ h + self.b_h)# hidden state activation function

            # store hidden state at this timestep
            self.last_hs[t] = h

            # final output (after processing whole sequence)
        y = self.Wh_o @ h + self.b_o

        # softmax to convert raw scores → probabilities
        pred = self.softmax(y)
        return pred

    def softmax(self, y):
        exp_x = np.exp(y - np.max(y))  # numerical stability trick
        return exp_x / np.sum(exp_x)  # probabilities sum to 1

    def loss(self,y_pred,y_actual):
        return -np.log(y_pred[ y_actual,0]) # provide loss (numerical term) 0 is for dimensional stability
                                            # if loss big -> high output
                                            # and if small-> low output

    def Bptt(self,d_o,learning_rate=0.01):#

        d_Wh_o  = d_o @ self.last_hs[len(self.last_inputs)-1].T# Gradient wrt W_hy is output error × hidden state at last timestep.
        d_b_o = d_o# d_o is loss wrt to output
        d_h = self.Wh_o.T @ d_o

        # store gradients accumulated over all timesteps
        d_Wi_h = np.zeros_like(self.Wi_h)
        d_Wh_h = np.zeros_like(self.Wh_h)
        d_b_h = np.zeros_like(self.b_h)

        #  Backpropagate through time (reverse order) Because RNN depends on previous states, so gradients must flow backward in time.
        for t in reversed(range(len(self.last_inputs))):# list store normally but we want to access reveres as lastest need to be accessed first
            h = self.last_hs[t]
            dh_raw = (1-h*h) * d_h # derivation of tan h = 1- h^2

            d_Wi_h += dh_raw @ self.last_inputs[t].T
            d_Wh_h += dh_raw @ self.last_hs[t-1].T
            d_b_h += dh_raw

            d_h = self.Wh_h.T @ dh_raw# Push error back into the previous hidden state (time t-1).

        for param, dparam in zip(
                [self.Wi_h,self.Wh_h,self.Wh_o,self.b_h,self.b_o],
                [d_Wi_h ,d_Wh_h ,d_Wh_o ,d_b_h,d_b_o]
        ):
            param -= self.learning_rate * dparam
            # param = the weight/bias you want to update
            # dparam = gradient of the loss wrt that parameter
            # self.learning_rate = step size (\eta)

def gen_data(list_len=3):
    X = [np.random.randint(0, 2) for _ in range(list_len)]
    Y = sum(X) % 2   # 0 = even, 1 = odd
    return [np.array([[x]]) for x in X], Y

rnn = RNN(input_size=1, hidden_size=16, output_size=2, learning_rate=0.05)

epochs = 1000
for epoch in range(epochs):
    X, y = gen_data()
    pred = rnn.forward(X)
    loss = rnn.loss(pred, y)

    d_y = pred.copy()
    d_y[y] -= 1 # replacement of one hot vector small trick (like onehotencoding)
    rnn.Bptt(d_y)

    if epoch % 50 == 0:
        prod = np.argmax(pred)
        print(f"Epoch {epoch}, Loss: {loss:.4f}, True: {y}, Pred: {prod}")
# Starter-ANN
ANN from scrach

1. Mini batch gradient descent

I have created a get_random_subset(X, y) function which takes the subset of the whole dataset


#for mini batch gradient descent random subset from X and y data points
def get_random_subset(X, y):
    rand_index = randint(0, len(X) - 1)
    X2 = np.array([X[rand_index]])
    y2 = np.array([y[rand_index]])
    #X = np.delete(X, X[rand_index], axis=0)
    #y = np.delete(y, y[rand_index])

    counter = 0
    while (counter < 9):
        rand_index = randint(0, len(X) - 1)
        X2 = np.append(X2, [X[rand_index]], axis=0)
        y2 = np.append(y2, [y[rand_index]], axis=0)

        #X = np.delete(X, X[rand_index], axis=0)
        #y = np.delete(y, y[rand_index])

        counter += 1
    return X2, y2

2. Annealing learning rate

I have used step decay approach which reduces the epsilon after every 5 epochs into half

#Annealing the learning rate by a half every 5 epochs(step decay)
        if (i % 5 == 0):
            Config.epsilon = Config.epsilon /2

3. Activation function

I have changed the tanh activation function for sigmoid function

I have created 2 functions to calculate the sigmoid and derivative of sigmoid.

#for sigmoid function
def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-1.0*z))
    return s

#for derivative of sigmoid function
def sigmoid_deravative(a1):
    a1 = a1 * (1 - a1)
    return a1

I have made changes in back propagation as, forward propagation, loss function and predict function

#Backpropagation for sigmoid
        delta3 = probs
        delta3[range(num_examples), y2] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        derivative = np.vectorize(sigmoid_deravative)
        delta2 = delta3.dot(W2.T)*derivative(a1)
        dW1 = np.dot(X2.T, delta2)
        db1 = np.sum(delta2, axis=0)

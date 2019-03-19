import numpy as np

"""
========================================================================================================================
Code for creating a MLP. Uses mini-batch SGD for training. Currently, activation is set to ReLU, with output being 
softmax. Hopefully, eventually add more activation modes and further improvements to training. 
========================================================================================================================
"""


def relu(x):
    """
    Runs ReLU activation on a given np Array
    :param x: An np array
    :return: Returns an np array as same dimension as x
    """
    return x * (x > 0)


def d_relu(x):
    """
    Calculate the derivative of a given np array, for ReLU.
    :param x: An np array
    :return:Returns an np array as same dimension as x
    """
    return 1 * (x > 0)


def softmax(x):
    """
    Calculates the softmax for a given np array. The softmax simplifies to the sigmoid case for binary classification.
    :param x: An input array
    :return: Ouput array with same dimensions as input array
    """
    # Prevent flow errors
    e_x = np.exp(x - np.max(x))
    soft = np.zeros((len(x), 2))
    # Individually create each value and calculate the partition function
    for value in range(len(e_x)):
        soft[value] = e_x[value] / np.sum(e_x[value])
    return soft


def binary_softmax_deriv(x):
    """
    This is the derivative of softmax with respect to binary classification. It is the same as the sigmoidal derivative.
    :param x: The values to take the derivative of
    :return: The derivative
    """
    return x * (1 - x)


class NeuralNetwork():
    """
    Creates a neural network with ReLU architecture and a softmax output layer
    """

    def __init__(self, input_params, labels, architecture=[32, 8], output_size=2):
        """

        :param input_params: The features matrix
        :param labels: An array of labels in one-hot encoded format
        :param architecture: The architecture of the hidden labels in the form of a list
        :param output_size: The total output size. This is equal to the number of class labels
        """

        self.X = input_params
        self.Y = labels
        self.layers = []
        self.stored_loss = []
        self.stored_accuracy = []
        self.epoch_count = 0
        # Sets an initial parameter for loss and accuracy. Not really needed but good coding practice
        self.loss = 1.0
        self.accurate = 0.5

        # Set up hidden layers
        for i in range(len(architecture)):
            if i == 0:
                # First layer needs to have shape dependent on number of input features
                hidden_layer = Layer(architecture[i], self.X.shape[1])
                self.layers.append(hidden_layer)
            else:
                hidden_layer = Layer(architecture[i], self.layers[-1].N)
                self.layers.append(hidden_layer)
        # Add output layer
        output_layer = OutputLayer(output_size, self.layers[-1].N)
        self.layers.append(output_layer)

    # TODO: Add dropout- this isn't getting done
    def train(self, epochs=500, batch_size=50, storage_rate=10, print_rate=20):
        """
        Trains a model based on the architecture and inputs specified above
        :param epochs: The total number of epochs to run. Default is 500
        :param batch_size: The size of each batch. Default is 50.
        :param storage_rate: How often values of loss and accuracy are stored. Defaut is every 10 epochs.
        :param print_rate: How often values of loss and accuracy are printed out. Default is every 20 epochs
        :return: Returns the stored loss and accuracy.
        """
        # Iterate up from each epoch while maintaining the total number of epochs. Has some weird behavior when running
        # a single epoch. Probably should not be 0-indexed.
        for epoch in range(self.epoch_count, self.epoch_count + epochs):
            self.epoch_count = epoch
            # Shuffle the samples at the start of each epcoh. This prevents some weird behaviors where weights become
            # dependent on location especially when batch_size/size(input) -> 1.
            shuffle = np.random.permutation(len(self.X))
            local_X = self.X[shuffle]
            local_Y = self.Y[shuffle]
            # Iterate across the batch_size. The total number of propagations is size(labels)/batch_size
            for batch_begin in range(0, len(self.X), batch_size):
                activation = self.layers[0].forward_propogation(local_X[batch_begin: batch_begin + batch_size])
                # Iterate through the remaining ReLU neurons
                for num in range(1, len(self.layers) - 1):
                    activation = self.layers[num].forward_propogation(activation)
                learn_fast = 0.001
                # Slow down the speed the deeper and deeper we can get into the model, especially as accuracy goes up
                if self.epoch_count >= 50 and self.accurate > 0.85:
                    # There's no basis in the literature of this, just needed something to slow down the speed that
                    # scales with accuracy and epoch count.
                    learn_fast = 1 / (1000 * (np.log(self.epoch_count) - 0.5))

                # Backprop the first layer
                self.layers[-1].backprop_adam(local_Y[batch_begin: batch_begin + batch_size], activation,
                                              self.epoch_count, learning_rate=learn_fast)
                # Backpropogation
                for num in reversed(range(0, len(self.layers) - 1)):
                    if num == len(self.layers) - 1:
                        prev_layer = self.layers[-1]
                    else:
                        prev_layer = self.layers[num + 1]
                    self.layers[num].backprop_adam(prev_layer, self.epoch_count, learning_rate=learn_fast)

            if epoch % storage_rate == 0:
                # Calculate the loss and accuracy for the entire data set. Stores it.
                self.accurate = self.accuracy()
                self.stored_loss.append(self.loss)
                self.stored_accuracy.append(self.accurate)
            if epoch % print_rate == 0:
                # NOTE: There's some odd behavior that occurs if print rate is smaller than storage rate.
                print("The current cost is", self.loss, 'for epoch', self.epoch_count)
                print("The current accuracy is", self.accurate)
        return self.stored_loss, self.stored_accuracy

    def accuracy(self):
        """
        Calculates the accuracy internally. Also, calculates loss on the entire data set.
        :return: Returns the total accuracy as a percentage.
        """
        prediction = self.predict(self.X)
        self.loss = self.layers[-1].loss_function(self.Y, prediction)
        # See better coding in classifier.
        total_valid = 0
        for value in range(len(prediction)):
            if np.array_equal(np.argmax(prediction[value]), np.argmax(self.Y[value])):
                total_valid += 1
        return (total_valid / len(prediction))

    def predict(self, x_test):
        """
        Predicts values given a feature matrix
        :param x_test: The feature matrix to predict answers on
        :return: Probability values of each sample being in a class. Size of array is n_samples x n_classes
        """
        activation = self.layers[0].forward_prop_predict(x_test)
        # Iterate through the remaining ReLU neurons
        for num in range(1, len(self.layers) - 1):
            activation = self.layers[num].forward_prop_predict(activation)
        prediction = self.layers[-1].forward_prop_predict(activation)
        return prediction


class Layer():
    """
    Creates a single layer of neurons
    """

    def __init__(self, size, n_input, activation=relu):
        """
        Initializes a hidden layer.
        :param size: The number of neurons.
        :param n_input: The size of the input.
        :param activation: Activation function to use. Default is ReLU.
        """
        self.n_input = n_input
        # Randomly initialize weight.
        self.W = np.random.uniform(low=-1 / n_input, high=1 / n_input, size=(n_input, size))
        self.B = np.zeros(size)
        self.N = size
        self.activation = activation
        # These are for Adam, whenever I get it to work.
        self.M = None
        self.V = None
        self.M_b = None
        self.V_b = None

    def forward_propogation(self, input):
        """
        Calculates an output given an input.
        :param input: The input matrix.
        :return: A matrix of the ouput weights. Equal to N.
        """
        self.input = input
        linear = np.matmul(input, self.W) + self.B
        # Store the derivative here for backprop
        self.d_activation = d_relu(linear)
        u1 = np.random.binomial(1, 0.6, size=linear.shape) / 0.6
        self.dropout=u1
        return (self.activation(linear)*u1)

    def forward_prop_predict(self, input):
        """
        When predicting either internally in accuracy or prediction, don't store values because this messes up backprop
        if we use it later.
        :param input:The input matrix
        :return:A matrix of the ouput weights. Equal to N.
        """
        linear = np.matmul(input, self.W) + self.B
        predicted = self.activation(linear)
        return predicted

    def backprop(self, prev_layer, learn_rate=0.0001):
        """
        Trains the data using mini-batch gradient descent.
        :param prev_layer: The previous layer. This is the layer that is closest to the output.
        :param learn_rate: The learning rate that we use. Default is 1e-4
        """
        # Here, the gradient is with respect to the cost function, thus we're trying to minimize that function
        error = self.d_activation * np.matmul(prev_layer.error, prev_layer.W.T)
        self.W -= learn_rate * np.matmul(self.input.T, error)
        self.B -= learn_rate * np.mean(error, axis=0)
        self.error = error

    def backprop_adam(self, prev_layer, time, beta_1=0.9, beta_2=0.999, learning_rate=0.001,
                      tolerance=1e-8):
        """

        :param prev_layer: The layer closet to this layer that is nearer the output layer. s
        :param time: The current time step- epoch count
        :param beta_1: The scaling factor for exponential decay of the first moment
        :param beta_2: The scaling factor for exponential decay of the second moment
        :param learning_rate: The learning rate
        :param tolerance: A tolerance amount such that we don't get underflow
        """
        error = self.d_activation * np.matmul(prev_layer.error, prev_layer.W.T)
        error *= self.dropout
        gradient = np.matmul(self.input.T, error)
        gradient_b = np.mean(gradient, axis=0)
        # If this is the first time that we are backprop
        if self.M is None:
            self.M = np.zeros_like(gradient)
            self.V = np.zeros_like(gradient)
            self.M_b = np.zeros_like(gradient_b)
            self.V_b = np.zeros_like(gradient_b)
        self.M = beta_1 * self.M + (1 - beta_1) * gradient
        self.V = beta_2 * self.V + (1 - beta_2) * np.power(gradient, 2)
        self.M_b = beta_1 * self.M_b + (1 - beta_1) * gradient_b
        self.V_b = beta_2 * self.V_b + (1 - beta_2) * np.power(gradient_b, 2)
        M_bias = self.M / (1 - beta_1 ** (time + 1))
        V_bias = self.V / (1 - beta_2 ** (time + 1))
        M_bias_b = self.M_b / (1 - beta_1 ** (time + 1))
        V_bias_b = self.V_b / (1 - beta_2 ** (time + 1))
        self.W -= learning_rate * M_bias / (np.sqrt(V_bias) + tolerance)
        self.B -= learning_rate * M_bias_b / (np.sqrt(V_bias_b) + tolerance)
        self.gradient = gradient
        self.gradient_b = gradient_b
        self.error = error


class OutputLayer():
    def __init__(self, size, n_input):
        """
        Initializes an output layer. Because this is a classifier, the activation is set to softmax.
        :param size: The number of classes.
        :param n_input: The size of the input.
        """
        self.n_input = n_input
        self.W = np.random.uniform(low=-1 / n_input, high=1 / n_input, size=(n_input, size))
        # Sanity check that W is the correct size.
        print(self.W)
        self.B = np.zeros(size)
        self.N = size
        # Values for Adam
        self.M = None
        self.V = None
        self.M_b = None
        self.V_b = None

    def forward_prop_predict(self, input):
        """
        When predicting either internally in accuracy or prediction, don't store values because this messes up backprop
        if we use it later.
        :param input:The input matrix
        :return:A matrix of the ouput weights. Equal to N.
        """
        linear = np.matmul(input, self.W) + self.B
        predicted = softmax(linear)
        return predicted

    def forward_propogation(self, input):
        """
        Calculates an output given an input.
        :param input: The input matrix.
        :return: A matrix of the ouput weights. Equal to N.
        """
        self.input = input
        linear = np.matmul(input, self.W) + self.B
        self.predicted = softmax(linear)
        # Still using ReLU because we pass this to the previous node
        self.d_activation = d_relu(input)
        return self.predicted

    # TODO: Figure out Adam
    def backprop(self, labels, input, learning_rate=0.001):
        """
        Calculates backprop for an output layer. Wraps the input method because both steps can be run at the same time.
        :param labels: One-hot encoded array containing the true labels
        :param input: Input from previous layer.
        :param learning_rate: The learning rate. Default is 0.0001
        """
        predicted = self.forward_propogation(input)
        # This is the derivative for binary softmax, using one-hot encoding.
        gradient = (predicted - labels) * binary_softmax_deriv(predicted)
        self.W -= learning_rate * np.matmul(input.T, gradient)
        self.B -= learning_rate * np.mean(gradient, axis=0)
        self.gradient = gradient

    def backprop_adam(self, labels, input, time, beta_1=0.9, beta_2=0.999, learning_rate=0.001, tolerance=1e-8):
        """
        Adam is an advancement to standard GD which uses the geometry of the curve to predict the update that is made during
        back propogation.
        :param labels: The labels for the current input
        :param input: The input from the previous layer
        :param time: The current time step- epoch count
        :param beta_1: The scaling factor for exponential decay of the first moment
        :param beta_2: The scaling factor for exponential decay of the second moment
        :param learning_rate: The learning rate
        :param tolerance: A tolerance amount such that we don't get underflow
        """
        predicted = self.forward_propogation(input)
        error = (predicted - labels) * binary_softmax_deriv(predicted)

        gradient = np.matmul(input.T, error)
        # Some mentions in literature use sum instead of mean
        gradient_b = np.mean(gradient, axis=0)
        # If this is the first time that we are backprop
        if self.M is None:
            self.M = np.zeros_like(gradient)
            self.V = np.zeros_like(gradient)
            self.M_b = np.zeros_like(gradient_b)
            self.V_b = np.zeros_like(gradient_b)
        self.M = beta_1 * self.M + (1 - beta_1) * gradient
        self.V = beta_2 * self.V + (1 - beta_2) * np.power(gradient, 2)
        self.M_b = beta_1 * self.M_b + (1 - beta_1) * gradient_b
        self.V_b = beta_2 * self.V_b + (1 - beta_2) * np.power(gradient_b, 2)
        M_bias = self.M / (1 - beta_1 ** (time + 1))
        V_bias = self.V / (1 - beta_2 ** (time + 1))
        M_bias_b = self.M_b / (1 - beta_1 ** (time + 1))
        V_bias_b = self.V_b / (1 - beta_2 ** (time + 1))
        self.W -= learning_rate * M_bias / (np.sqrt(V_bias) + tolerance)
        self.B -= learning_rate * M_bias_b / (np.sqrt(V_bias_b) + tolerance)
        self.gradient = gradient
        self.gradient_b = gradient_b
        self.error = error

    # The loss function is only dependent on values from the output layer.
    def loss_function(self, labels, predicted=None):
        """
        Calculates the training loss of our prediction. Larger values indicate larger error. SGD/Adam attempts to minimize
        this function.
        :param labels: The true values for the data set.
        :param predicted: The one-hot encoded prediction from the neural network. Default is None.
        :return: The cross-entropy, aka training loss
        """
        # If we don't specify the prediction, assume that it's the current prediction.
        if predicted is None:
            predicted = self.predicted
        total_loss = np.zeros(len(labels))
        # Loss for binary classifier is defined as -log(P(prediction=true_class))
        for i in range(len(labels)):
            if np.argmax(labels[i]) == 1:
                total_loss[i] = -np.log(predicted[i][1])
            else:
                total_loss[i] = -np.log(predicted[i][0])
        # Some literature uses summation, more descriptive to use average (i.e easier to calculate how accurate your
        # model currently is)
        cross_entropy = np.average(total_loss)

        return cross_entropy

import math
import numpy
import random
import pandas
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, inputs, hidden, outputs):
        """
        Initializes the neural network object
        :param inputs: number of inputs
        :param hidden: number of hidden neurons
        :param outputs: number of output neurons
        """
        # Specify activations and number of units in each layer
        self.inputs, self.hidden, self.outputs = inputs + 1, hidden, outputs
        self.ai, self.ah, self.ao = numpy.ones(self.inputs), numpy.ones(self.hidden), numpy.ones(self.outputs)
        # Create very small weight matrices using numpy random matrices
        self.wi = numpy.random.uniform(-0.01, 0.01, (self.inputs, self.hidden))
        self.wo = numpy.random.uniform(-0.01, 0.01, (self.inputs, self.outputs))
        # Create matrices for the BP momentum zero = no bias
        self.ci = numpy.zeros((self.inputs, self.hidden))
        self.co = numpy.zeros((self.hidden, self.outputs))

    def forward_pass(self, inputs):
        """
        This method passed the inputs through the network to get the output
        :param inputs: the input values to pass through (forward pass)
        :return: the output for the given inputs
        """
        if len(inputs) != self.inputs - 1:
            raise ValueError('wrong number of inputs')
        # input activations
        for i in range(self.inputs - 1):
            self.ai[i] = inputs[i]
        # hidden activations
        for j in range(self.hidden):
            sum = 0.0
            for i in range(self.inputs):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = self.sigmoid(sum)
        # output activations
        for k in range(self.outputs):
            sum = 0.0
            for j in range(self.hidden):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = self.sigmoid(sum)

        return self.ao[:]

    # our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
    def sigmoid(self, x):
        return math.tanh(x)

    # derivative of our sigmoid function, in terms of the output (i.e. y)
    def dsigmoid(self, y):
        return 1.0 - y ** 2

    def back_propagation(self, targets, n, m):
        """
        This method updates the neural network weights using Adjoint Differentiation
        :param targets: what we expected out
        :param n: the learning rate
        :param m: the momentum rate
        :return: the sum squared error of the network
        """
        if len(targets) != self.outputs:
            raise ValueError('wrong number of target values')
        # Output layer errors
        deltas = [0.0] * self.outputs
        for k in range(self.outputs):
            error = targets[k] - self.ao[k]
            deltas[k] = self.dsigmoid(self.ao[k]) * error
        # Hidden layer errors
        hidden_deltas = [0.0] * self.hidden
        for j in range(self.hidden):
            error = 0.0
            for k in range(self.outputs):
                error += deltas[k] * self.wo[j][k]
            hidden_deltas[j] = self.dsigmoid(self.ah[j]) * error
        # Update the output weights
        for j in range(self.hidden):
            for k in range(self.outputs):
                weight_update = deltas[k] * self.ah[j]
                self.wo[j][k] += n * weight_update + m * self.co[j][k]
                self.co[j][k] = weight_update
        # Update input weights
        for i in range(self.inputs):
            for j in range(self.hidden):
                weight_update = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] += n * weight_update + m * self.ci[i][j]
                self.ci[i][j] = weight_update
        # Calculate the error - this is a tailor series expansion (derivative)
        error = 0.0
        for k in range(len(targets)):
            error += math.pow(targets[k] - self.ao[k], 2.0)
        return error

    def weights(self):
        """
        This prints out the weights for introspection
        """
        print('Input weights:', pandas.DataFrame(self.wi))
        print('Output weights:', pandas.DataFrame(self.wo))

    def get_classifications(self, patterns):
        """
        This gets the outputs for a set of input patterns (for testing accuracy)
        :param patterns: the input patterns
        :return: the output results
        """
        classifications = []
        for p in patterns:
            out = self.forward_pass(p[0])[0]
            classifications.append(round(out, 0))
        return classifications

    def train(self, patterns, iterations=250):
        """
        This method trains the neural network i.e. iterated back-propagation
        :param patterns: the input patterns
        :param iterations: max iterations (this method has early stopping)
        :param n: the learning rate
        :param m: the momentum rate
        """
        # n is the learning rate
        # m us the momentum factor
        fitness = []
        errors = [float('+inf')]
        wo_best, wi_best = self.wo, self.wi
        plt.ion()
        plt.figure(figsize=(10, 10))
        self.plot_grid("weights/initial", self.wi, self.wo, [])
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                self.forward_pass(numpy.array(p[0]))
                error += self.back_propagation(p[1], 0.7, 0.1)
            errors.append(error)
            percent = round(i / iterations * 100, 3)
            if error == min(errors):
                wo_best, wi_best = wo_best, wi_best
            if percent % 1.0 == 0:
                print(percent, "%\t", error)
                fitness.append(error)
                s = "weights/" + str(i).zfill(6)
                self.plot_grid(s, wi_best, wo_best, errors)
            if random.random() < 0.1:
                self.wo += numpy.random.uniform(-1.0, 1.0, (self.inputs, self.outputs))
                self.wi += numpy.random.uniform(-1.0, 1.0, (self.inputs, self.outputs))
                jump_error = 0.0
                for p in patterns:
                    self.forward_pass(numpy.array(p[0]))
                    jump_error += self.back_propagation(p[1], 0.7, 0.1)
                if jump_error > error:
                    self.wo, self.wi = wo_best, wi_best
        self.wo, self.wi = wo_best, wi_best
        plt.plot(fitness)
        plt.show()

    def plot_grid(self, name, wm_i, wm_o, errors):
        # Cool colour map found here - http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
        fig1 = plt.subplot(311)
        fig2 = plt.subplot(312)
        fig3 = plt.subplot(313)

        fig1.matshow(wm_i.transpose(), cmap="PRGn")
        fig2.matshow(wm_o.transpose(), cmap="PRGn")
        fig3.cla()
        fig3.plot(errors[len(errors)-100:])

        plt.savefig(name + '.png')
        plt.draw()

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.forward_pass(p[0]))


def get_patterns(file_name):
    """
    This method download the patterns i.e. credit data
    :param file_name: the file containing the patterns / data
    :return: the list of patterns in the form [[inputs],[outputs]]
    """
    pattern_data = pandas.read_csv(file_name)
    pattern_inputs = pattern_data.drop("Target", 1)
    pattern_target = pattern_data["Target"]
    patterns = []
    for p in range(len(pattern_inputs)):
        p_input = list(pattern_inputs.iloc[p])
        p_target = [pattern_target.iloc[p]]
        patterns.append([p_input, p_target])
    return patterns, list(pattern_target)


def test_network(n, patterns, targets, label):
    """
    This just determines the accuracy of the network
    :param n: the neural network
    :param patterns: the input patterns in form [[inputs],[outputs]]
    :param targets: the expected outputs [outputs]
    :param label: the name of the data set
    """
    correct = 0
    classes_in = list(targets)
    classes_out = n.get_classifications(patterns)
    for i in range(len(classes_out)):
        # print('%.4f' % classes_out[i], '%.4f' % classes_in[i])
        if classes_in[i] == classes_out[i]:
            correct += 1
    print("Accuracy on", label, "set =", float(correct / len(classes_out)) * 100, "%")


def main():
    """
    Main method for testing the neural network
    """
    train, train_targets = get_patterns("#TrainingSet.csv")
    test, test_targets = get_patterns("#TestingSet.csv")
    neural_network = NeuralNetwork(24, 1, 1)
    neural_network.train(train)
    test_network(neural_network, train, train_targets, "Training")
    test_network(neural_network, test, test_targets, "Testing")
    neural_network.weights()


def main_two():
    """
    Main method for testing the neural network
    """
    train, train_targets = get_patterns("#TrainingSetTwo.csv")
    test, test_targets = get_patterns("#TestingSetTwo.csv")
    neural_network = NeuralNetwork(15, 4, 1)
    neural_network.train(train)
    test_network(neural_network, train, train_targets, "Training")
    test_network(neural_network, test, test_targets, "Testing")
    neural_network.weights()


if __name__ == '__main__':
    main_two()
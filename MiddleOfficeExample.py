import math
import numpy
import random
import pandas
import cProfile
from scipy.special import expit
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
        self.wi = numpy.random.random((self.inputs, self.hidden))/1000
        self.wo = numpy.random.random((self.hidden, self.outputs))/1000
        # Create matrices for the BP momentum zero = no bias
        self.ci = numpy.zeros((self.inputs, self.hidden))
        self.co = numpy.zeros((self.hidden, self.outputs))

    def forward_pass(self, inputs):
        """
        This method passed the inputs through the network to get the output
        :param inputs: the input values to pass through (forward pass)
        :return: the output for the given inputs
        """
        if len(inputs) != self.inputs-1:
            raise ValueError('wrong number of inputs')
        # Linear functions for the inputs
        for i in range(self.inputs-1):
            self.ai[i] = inputs[i]
        # Sigmoid functions for the hidden layer
        for j in range(self.hidden):
            # Weighted sum of inputs * weights
            weight_sum = self.ai * self.wi[:, j]
            self.ah[j] = math.tanh(weight_sum.sum())
        # Sigmoid functions for the output layer
        for k in range(self.outputs):
            # Weighted sum of hidden * weights
            weight_sum = self.ah * self.wo[:, k]
            self.ao[k] = math.tanh(weight_sum.sum())
        # Return the outputs
        return self.ao[:]

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
        errors = targets - self.ao
        deltas = (1.0 - self.ao**2) * errors
        # Hidden layer errors
        hidden_deltas = numpy.zeros(self.hidden)
        for j in range(self.hidden):
            error = deltas * self.wo[j, :]
            hidden_deltas[j] = (1.0 - self.ah[j]**2) * error
        # Update the output weights
        for j in range(self.hidden):
            for k in range(self.outputs):
                weight_update = deltas[k] * self.ah[j]
                self.wo[j][k] += n * weight_update + m * self.co[j][k]
                self.co[j][k] = weight_update
        # Update input weights
        for i in range(self.inputs):
            for j in range(self.hidden):
                weight_update = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] += n * weight_update + m * self.ci[i][j]
                self.ci[i][j] = weight_update
        # Calculate the error - this is a tailor series expansion (derivative)
        errors = 0.5 * (targets - self.ao)**2
        return errors.sum()

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

    def train(self, patterns, iterations=250, n=0.75, m=0.1):
        """
        This method trains the neural network i.e. iterated back-propagation
        :param patterns: the input patterns
        :param iterations: max iterations (this method has early stopping)
        :param n: the learning rate
        :param m: the momentum rate
        """
        # n is the learning rate
        # m us the momentum factor
        mutation_rate = 0.25
        errors = [float('+inf')]
        wo_best, wi_best = self.wo, self.wi
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                self.forward_pass(numpy.array(p[0]))
                error += self.back_propagation(p[1], n, m)
            print(round(i / iterations * 100, 2), "%\t", error)
            if random.random() < mutation_rate:
                print("This is a mutation step")
                old_wi, old_wo = self.wi, self.wo
                self.wi = numpy.random.random((self.inputs, self.hidden))/100
                self.wo = numpy.random.random((self.hidden, self.outputs))/100
                error_mutation = 0.0
                for p in patterns:
                    self.forward_pass(numpy.array(p[0]))
                    error_mutation += self.back_propagation(p[1], n, m)
                if error < error_mutation:
                    self.wi, self.wo = old_wi, old_wo
            errors.append(error)
            if error == min(errors):
                wo_best, wi_best = self.wo, self.wi
        self.wo, self.wi = wo_best, wi_best
        plt.plot(errors)
        plt.show()

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
    print("Accuracy on", label, "set =", float(correct/len(classes_out))*100, "%")


def main():
    """
    Main method for testing the neural network
    """
    train, train_targets = get_patterns("#TrainingSet.csv")
    test, test_targets = get_patterns("#TestingSet.csv")
    neural_network = NeuralNetwork(24, 10, 1)
    neural_network.train(train)
    test_network(neural_network, train, train_targets, "Training")
    test_network(neural_network, test, test_targets, "Testing")
    neural_network.weights()


if __name__ == '__main__':
    main()
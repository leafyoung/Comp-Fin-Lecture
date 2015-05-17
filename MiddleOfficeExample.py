import math
import numpy
import pandas
import cProfile


class NeuralNetwork:
    def __init__(self, inputs, hidden, outputs):
        # Specify activations and number of units in each layer
        self.inputs, self.hidden, self.outputs = inputs + 1, hidden, outputs
        self.ai, self.ah, self.ao = numpy.ones(self.inputs), numpy.ones(self.hidden), numpy.ones(self.outputs)
        # Create very small weight matrices using numpy random matrices
        self.wi = numpy.random.random((self.inputs, self.hidden))/100
        self.wo = numpy.random.random((self.hidden, self.outputs))/100
        # Create matrices for the BP momentum zero = no bias
        self.ci = numpy.zeros((self.inputs, self.hidden))
        self.co = numpy.zeros((self.hidden, self.outputs))

    def update(self, inputs):
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
        # calculate error
        errors = 0.5 * (targets - self.ao)**2
        return errors.sum()

    def weights(self):
        print('Input weights:', pandas.DataFrame(self.wi))
        print('Output weights:', pandas.DataFrame(self.wo))

    def get_classifications(self, patterns):
        classifications = []
        for p in patterns:
            out = self.update(p[0])[0]
            classifications.append(round(out, 0))
        return classifications

    def train(self, patterns, iterations=1000, n=0.75, m=0.1):
        # n is the learning rate
        # m us the momentum factor
        errors = []
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(numpy.array(inputs))
                error = error + self.back_propagation(targets, n, m)
            errors.append(error)
            # Early stopping condition
            if len(errors) > 10:
                limit = 0.001
                # This prevents the updates becoming chaotic!
                improvement = (errors[len(errors) - 10] / errors[i]) - 1
                if improvement < limit:
                    break


def get_patterns(file_name):
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
    correct = 0
    classes_in = list(targets)
    classes_out = n.get_classifications(patterns)
    for i in range(len(classes_out)):
        if classes_in[i] == classes_out[i]:
            correct += 1
    print("Accuracy on", label, "set =", float(correct/len(classes_out))*100, "%")


def main():
    train, train_targets = get_patterns("#TrainingSet.csv")
    test, test_targets = get_patterns("#TestingSet.csv")
    neural_network = NeuralNetwork(24, 5, 1)
    neural_network.train(train)
    test_network(neural_network, train, train_targets, "Training")
    test_network(neural_network, test, test_targets, "Testing")
    neural_network.weights()

if __name__ == '__main__':
    main()
import math
import numpy
import random
import numpy.random as nrand
import matplotlib.pyplot as plt


def ornstein_uhlenbeck(t, r, mu, a, delta, sigma):
    """
    Used to model interest rates. Ornstein Uhlenbeck is mean-reverting.
    :param t: length of the simulated rates
    :param r: starting value of the rates
    :param mu: mean interest rate over t
    :param a: rate of mean-reversion
    :param delta: instantaneous time
    :param sigma: volatility of the rates
    :return: numpy array of t interest rates
    """
    ou_levels = [r]
    # An array of t brownian motion returns (normally distributed returns)
    brownian_motion = nrand.normal(loc=0, scale=math.sqrt(delta) * sigma, size=t)
    for i in range(1, t):
        drift = a * (mu - ou_levels[i - 1]) * delta
        ou_levels.append(ou_levels[i - 1] + drift + brownian_motion[i - 1])
    return numpy.array(ou_levels)


def model_calibrator(iterations, x_list, y_list):
    """
    Calibrates the Ornstein-Uhlenbeck process using t-1 (x_list) and t rates (y_list)
    :param iterations: number of iterations
    :param x_list: t-1 interest rates
    :param y_list: t interest rates
    :return: calibration array
    """
    x = nrand.uniform(0.0, 1.0, 2)
    for i in range(iterations):
        # Step size decreases over time but it bounded from below by 0.05
        step = max(1/(i+1), 0.05)
        # Generate a new solution (greedy)
        x, sse = neighbourhood(x, x_list, y_list, step)
        # Print the solution to console
        print(i, x, sse)
    return x


def neighbourhood(s, x_list, y_list, step, num=20):
    """
    Update the solution, s, with a better neighbour
    :param s: current solution
    :param x_list: t-1 interest rates
    :param y_list: t interest rates
    :param step: size of neighbourhood function
    :param num: number of neighbours to generate
    :return: the best neighbour found
    """
    # Calculate the current fitness
    x, y, xx, yy, xy, sse = regression(s, x_list, y_list)
    best_x, best_sse = s, sse
    # Generate num neighbours
    for i in range(num):
        # Add +- step to each dimension of best_x
        neighbour = best_x + nrand.uniform(-step, step, 2)
        x, y, xx, yy, xy, sse = regression(neighbour, x_list, y_list)
        # If the neighbours fitness is better, update the best x
        if sse < best_sse:
            best_x = neighbour
            best_sse = sse
    return best_x, best_sse


def regression(s, x_list, y_list):
    """
    Vectorized function for calculation regression fitness values
    :param s: the current solution
    :param x_list: t-1 interest rates
    :param y_list: t interest rates
    :return: fitness quantities
    """
    # XX list is X^2, YY list is Y^2, XY list is X*Y
    xx_list, yy_list, xy_list = x_list**2, y_list**2, x_list * y_list
    # Calculate the offsets from the solution (line) to the x's and y's
    y_offset = numpy.abs(((x_list * s[0]) + s[1]) - y_list)
    x_offset = numpy.abs(((y_list - s[1]) / s[0]) - x_list)
    # Multiply the x and y offsets together
    xy_offsets = x_offset * y_offset
    # Sum up the different quantities
    x, y = x_list.sum(), y_list.sum()
    xx, yy = xx_list.sum(), yy_list.sum()
    xy, sse = xy_list.sum(), xy_offsets.sum()
    return x, y, xx, yy, xy, sse


def plot(s, x_list, y_list):
    """
    Method for plotting the points + line
    :param s: current solution
    :param x_list: t-1 interest rates
    :param y_list: t interest rates
    """
    plt.style.use(['bmh'])
    min_x = min(x_list)
    max_x = max(x_list)
    xs = numpy.linspace(min_x, max_x, 100)
    line = []
    for x in xs:
        line.append(s[0] * x + s[1])
    plt.scatter(x_list, y_list)
    plt.plot(xs, line, 'k')
    plt.show()


def main():
    days, delta = 1000, float(1/252)
    mu_start, a_start, sigma_start = random.random(), random.randint(1, 12), random.random()
    rates = ornstein_uhlenbeck(days, mu_start, mu_start, a_start, delta, sigma_start)
    # Take all the rates and split into t and t-1
    rates_minus = rates[:len(rates) - 1:]
    rates_plus = rates[1::]
    # Find a calibration for the model using hill climbing
    s = model_calibrator(250, rates_plus, rates_minus)
    plot(s, rates_plus, rates_minus)
    # This is some fancy maths for relating the ornstein uhlenbeck model parameters to the co-efficients
    # in the regression - http://www.sitmo.com/article/calibrating-the-ornstein-uhlenbeck-model/
    x, y, xx, yy, xy, sse = regression(s, rates_plus, rates_minus)
    a = ((days * xy) - (x * y)) / (days * xx - x**2)
    b = (y - a * x) / days
    p = a * (days * xy - x * y)
    var = (days * yy - y**2 - p)/(days * (days - 2))
    l = - (numpy.log(a) / delta)
    mu = b / (1 - a)
    sigma = math.sqrt(var) * math.sqrt((-2 * numpy.log(a))/(delta * (1 - a**2)))
    # Simulate a new process using the results to see how they compare
    rates_sim = ornstein_uhlenbeck(days, mu, mu, l, delta, sigma)
    plt.plot(rates_plus)
    plt.plot(rates_sim)
    plt.show()
    # Print the regressed parameters
    print("Actual parameters", "mu=", mu_start, "a=", a_start, "sigma=", sigma_start)
    print("Regressed parameters", "mu=", mu, "a=", l, "sigma=", sigma)

if __name__ == "__main__":
    main()
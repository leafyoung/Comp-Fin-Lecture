import math
import numpy
import pandas
import Quandl
import cProfile
import numpy.random as nrand
import matplotlib.pyplot as plt
import scipy.optimize as opt


class Swarm:
    def __init__(self, n, i, dim, p, c1=1.496180, c2=1.496180, w=0.729844, restart=True):
        """
        Initialization method for the swarm
        :param n: number of particles
        :param i: number of iterations
        :param dim: dimensionality
        :param c1: social component constant
        :param c2: cognitive component constant
        :param w: inertia constant
        :param p: problem object
        """
        self.iterations = i
        # Set problem specific variables
        self.swarm, self.p, self.dim = [], p, dim
        # Set particle swarm optimization parameters
        self.c1, self.c2, self.w = c1, c2, w
        # Create a swarm of particles
        for i in range(n):
            self.swarm.append(Particle(dim, c1, c2, w, p))
        # Variables for the GBest optimizer
        self.bounds = []
        for i in range(dim):
            self.bounds.append((p.bounds[0], p.bounds[1]))
        # Specify the equality constraint i.e. sum(x) == 1.0
        # self.constr = ({'type': 'eq', 'fun': self.equality})
        self.restart = restart

    def optimize(self):
        """
        This method runs the optimization algorithm
        :return: the best solution (vector) and it's fitness
        """
        # Memory of the best seen solution, and all historical solutions
        fits, best_g_best, best_g_best_fit, p_gbest = [], None, float('+inf'), None
        for i in range(self.iterations):
            # Get the best solution in the swarm
            g_best, f, avg_f = self.get_best()
            # Append the avg swarm fitness
            fits.append(avg_f)
            # If this g best is better than the best g best
            if f < best_g_best_fit:
                # Update the best g best
                best_g_best = g_best
                best_g_best_fit = f
            # For each particle
            for particle in self.swarm:
                # If the particle is not the g best
                if particle is not g_best:
                    # Update the particle
                    particle.update(g_best.x)
                    # Satisfy the equality constraint
                    particle.x = self.p.repair(particle.x)
                    distance = (particle.x - g_best.x) ** 2
                    # If set to restart converged particles and particle == gbest then
                    if self.restart and numpy.sum(distance) < 0.1:
                        # Re-initialize the particle solution somewhere in the search space
                        particle.x = nrand.uniform(self.p.bounds[0], self.p.bounds[1], self.dim)
                        # Satisfy the equality constraint sum(x) == 1.0
                        particle.x = self.p.repair(particle.x)
                # If this particle is the g best
                elif g_best is not p_gbest:
                    # Optimize the g best particle using a traditional optimization algorithm
                    res = opt.minimize(self.p.fit, particle.x, bounds=self.bounds)
                    particle.x = res.x
                    # pass
            # Early stopping
            '''
            if i > 100:
                # Calculate change in average swarm fitness
                change = (fits[i] / fits[i - 100]) - 1
                # If this is small then break / stop
                if -0.0001 <= change <= 0.0001:
                    break
            '''
            p_gbest = g_best
            if i % 100 == 0:
                print(best_g_best_fit, best_g_best.x)
        # Return the best solution
        return best_g_best.x, best_g_best_fit

    def equality(self, x):
        return sum(x) - 1.0

    def get_best(self):
        """
        This method returns the best particle in the swarm
        :return: best particle, fitness, average fitness
        """
        g_best = None
        g_best_fit = float('+inf')
        fits = []
        for particle in self.swarm:
            fit = self.p.fit(particle.x)
            fits.append(fit)
            if fit < g_best_fit:
                g_best = particle
                g_best_fit = fit
        nfits = numpy.array(fits)
        return g_best, g_best_fit, nfits.mean()


class Particle:
    def __init__(self, dim, c1, c2, w, p):
        """
        Initialization method for a particle in the swarm
        :param dim: dimensionality of the problem
        :param c1: social component constant
        :param c2: cognitive component constant
        :param w: inertia constant
        :param p: problem object
        """
        self.dim, self.c1, self.c2, self.w, self.p = dim, c1, c2, w, p
        # Initialize the solution randomly in the search space
        self.x = nrand.uniform(p.bounds[0], p.bounds[1], dim)
        self.v = nrand.uniform(0.000, 0.001, dim)
        self.p_best, self.p_best_fit = self.x, float('-inf')

    def update(self, g_best):
        """
        Update rule for a particle given the g best solution
        :param g_best: the best solution in the swarm
        """
        c1r1 = nrand.uniform(0, 1, self.dim) * self.c1
        c2r2 = nrand.uniform(0, 1, self.dim) * self.c2
        cog = (self.x - self.p_best) * c1r1
        soc = (self.x - g_best) * c2r2
        self.v += cog + soc
        self.x += self.v * self.w
        # Update the personal best position
        self.x = self.p.repair(self.x)
        if self.p.fit(self.x) > self.p_best_fit:
            self.p_best = self.x


class WilsonProblem:
    def __init__(self):
        self.initial = [0.2, 0.3, 0.4]
        self.data = [5, 9, 8, 4, 7]
        self.bounds = [0.01, 0.99]
        # print(self.em(self.initial, self.x))

    def fit(self, x):
        # print(x)
        lam = x[0]
        p = x[1]
        q = x[2]
        s = 0
        for index in range(len(self.data)):
            s += math.log(lam * math.pow(p, self.data[index]) *
                          math.pow((1 - p), (10 - self.data[index])) +
                          (1 - lam) * math.pow(q, self.data[index]) *
                          math.pow((1 - q), (10 - self.data[index])))
        return -s

    def repair(self, x):
        for i in range(len(x)):
            if x[i] < 0.01:
                x[i] = 0.01
            elif x[i] > 0.99:
                x[i] = 0.99
        x[0] = 0.5
        return x


def main():
    """
    Main method for portfolio optimization example
    """
    # dim, p, c1=1.496180, c2=1.496180, w=0.729844, restart=True
    problem = WilsonProblem()
    swarm = Swarm(50, 1000, 3, problem)
    best_x, best_f = swarm.optimize()
    print(best_x, best_f)


if __name__ == '__main__':
    main()
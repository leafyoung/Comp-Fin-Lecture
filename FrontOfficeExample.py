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
        self.constr = ({'type': 'eq', 'fun': self.equality})
        self.restart = restart

    def optimize(self):
        """
        This method runs the optimization algorithm
        :return: the best solution (vector) and it's fitness
        """
        # Memory of the best seen solution, and all historical solutions
        fits, best_g_best, best_g_best_fit = [], None, float('-inf')
        for i in range(self.iterations):
            # Get the best solution in the swarm
            g_best, f, avg_f = self.get_best()
            # Append the avg swarm fitness
            fits.append(avg_f)
            # If this g best is better than the best g best
            if f > best_g_best_fit:
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
                    distance = (particle.x - g_best.x)**2
                    # If set to restart converged particles and particle == gbest then
                    if self.restart and numpy.sum(distance) < 0.1:
                        # Re-initialize the particle solution somewhere in the search space
                        particle.x = nrand.uniform(self.p.bounds[0], self.p.bounds[1], self.dim)
                        # Satisfy the equality constraint sum(x) == 1.0
                        particle.x = self.p.repair(particle.x)
                # If this particle is the g best
                else:
                    # Optimize the g best particle using a traditional optimization algorithm
                    res = opt.minimize(self.p.fit_inv, particle.x, bounds=self.bounds, constraints=self.constr)
                    particle.x = res.x
            # Early stopping
            if i > 100:
                # Calculate change in average swarm fitness
                change = (fits[i] / fits[i-100]) - 1
                # If this is small then break / stop
                if -0.001 <= change <= 0.001:
                    break
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
        g_best_fit = float('-inf')
        fits = []
        for particle in self.swarm:
            fit = self.p.fit(particle.x)
            fits.append(fit)
            if fit > g_best_fit:
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
        if self.p.fit(self.x) > self.p_best_fit:
            self.p_best = self.x


class Portfolio:
    def __init__(self, bounds, rets, risk, corr):
        """
        Initializes a portfolio (basically the problem)
        :param bounds: min and max weight for each asset
        :param rets: the expected returns per asset
        :param risk: the expected risk per asset
        :param corr: the correlation matrix
        """
        self.bounds = bounds
        self.er = rets
        self.es = risk
        self.c = corr

    def fit(self, x):
        """
        This calculate the fitness of the solution - quite slow
        :param x: the weights of the solution
        :return: the sharpe ratio
        """
        r = numpy.sum(self.er * x)
        risk = numpy.sum(self.es**2 * x**2)
        for i in range(len(x)):
            # i th row of the corr matrix
            corr_i = numpy.array(self.c[i][:])
            risk_i = x * x[i] * self.es * self.es[i] * corr_i
            risk += risk_i.sum()
        sharpe = r / math.sqrt(risk)
        return sharpe

    def fit_inv(self, x):
        return -self.fit(x)

    def repair(self, x):
        """
        This method satisfies the equality constraint
        :param x: the weights of the solution
        :return: the 'normalized' solution
        """
        norm = x / numpy.sum(x)
        for i in range(len(norm)):
            norm[i] = max(norm[i], self.bounds[0])
            norm[i] = min(norm[i], self.bounds[1])
        norm_x = norm / numpy.sum(norm)
        return norm_x


class Fetcher:
    def __init__(self, quandl_auth):
        """
        Initializes a fetcher object for downloading data
        :param quandl_auth: my quandl auth token
        """
        self.token = quandl_auth

    def get_stock(self, ticker, start, end, drop=None, collapse="daily", transform="rdiff"):
        """
        :param ticker: ticker
        :param start: start-date
        :param end: end-date
        :param drop: columns in data-frame to drop
        :param collapse: frequency of data
        :param transform: rdiff = simple returns
        """
        quandl_args = "GOOG/" + ticker
        if drop is None:
            drop = ["Open", "High", "Low", "Volume"]
        hash_val = str(ticker) + "_" + str(start) + "_" + str(end)
        try:
            cached_data = pandas.read_csv("cache/" + str(hash_val))
            cached_data = cached_data.set_index("Date")
            return cached_data
        except IOError:
            try:
                # print("Downloading", data_set)
                # Otherwise download the data frame from scratch
                if transform is not "None":
                    downloaded_data_frame = Quandl.get(quandl_args, authtoken=self.token, trim_start=start,
                                                       trim_end=end, collapse=collapse, transformation=transform)
                else:
                    downloaded_data_frame = Quandl.get(quandl_args, authtoken=self.token, trim_start=start,
                                                       trim_end=end, collapse=collapse)
                # Remove any unnecessary columns and rename the columns
                # print downloaded_data_frame.columns
                updated_column_labels = []
                for column_label in downloaded_data_frame.columns:
                    if column_label in drop:
                        downloaded_data_frame = downloaded_data_frame.drop([column_label], axis=1)
                    else:
                        updated_column_labels.append(quandl_args + "_" + column_label)
                downloaded_data_frame.columns = updated_column_labels
                downloaded_data_frame.to_csv("cache/" + str(hash_val))
                return downloaded_data_frame
            except Quandl.DatasetNotFound:
                print("Exception - DataSetNotFound", quandl_args)
            except Quandl.CodeFormatError:
                print("Exception - CallFormatError", quandl_args)
            except Quandl.DateNotRecognized:
                print("Exception - DateNotRecognized", quandl_args)
            except Quandl.ErrorDownloading:
                print("Exception - ErrorDownloading", quandl_args)
            except Quandl.ParsingError:
                print("Exception - ParsingError", quandl_args)

    def get_stocks(self, tickers, start, end, drop=None, collapse="daily", transform="rdiff"):
        """
        A wrapper for get_stock that takes in a list of tickers and joins the resulting data-frames together
        """
        all_data_sets = None
        for ticker in tickers:
            downloaded_data_frame = self.get_stock(ticker, start, end, drop, collapse, transform)
            if all_data_sets is None:
                all_data_sets = downloaded_data_frame
            else:
                if downloaded_data_frame is not None:
                    if not downloaded_data_frame.empty:
                        all_data_sets = all_data_sets.join(downloaded_data_frame, how="outer")
        return all_data_sets


def main():
    """
    Main method for portfolio optimization example
    """
    # The tickers of the stocks in our portfolio (10 stocks)
    tickers = list(pandas.read_csv("#PortfolioStocks.csv")["Ticker"])
    start_date, end_date = "2010-01-01", "2015-05-01"
    fetcher = Fetcher("N9HccV672zuuiU5MUvcq")
    # This uses the Fetcher to download all the data
    data = fetcher.get_stocks(tickers, start_date, end_date)
    # Create a benchmark portfolio (equal weights)
    equal = numpy.empty(len(data.columns))
    equal.fill(float(1/len(data.columns)))
    # Create out optimal portfolio
    optimized_weights = equal
    # List of returns for each portfolio
    eq, opt = [], []
    # Starting date components
    start, ym, mm, dm = 0, "2010", "01", "01"
    # For each datum in the data-frame (daily returns)
    for i in range(len(data)):
        # Get the date components
        dd = str(data.index[i]).split('-')
        y, m, d = dd[0], dd[1], dd[2]
        # If we are in a new month and the month is either June / December
        if m != mm and int(m) % 6 == 0:
            # Slice the data-frame between the last date and this date
            data_slice = data.iloc[start:i]
            # Calculate the inputs to the portfolio
            risk = numpy.array(data_slice.std())
            corr = numpy.matrix(data_slice.corr())
            rets = numpy.array(data_slice.mean())
            # Calculate the returns for each portfolio
            eq_ret = float(numpy.sum(rets * equal))
            opt_ret = float(numpy.sum(rets * optimized_weights))
            # Append the returns to the list
            eq.append(eq_ret)
            opt.append(opt_ret)
            # Optimize the portfolio for the next slice
            p = Portfolio([-0.25, 0.25], rets, risk, corr)
            s = Swarm(25, 5000, len(data.columns), p)
            optimized_weights, f = s.optimize()
            # Print out the results from this slice
            print(d, m, y, eq_ret, opt_ret, f)
            ym, mm, dm = y, m, d
            start = i
    # Calculate the compounded returns (of the log returns)
    eq_c, opt_c = [1.0], [1.0]
    for i in range(1, len(eq)):
        eq_c.append(eq_c[i-1] * math.exp(math.log(eq[i-1] + 1)))
        opt_c.append(opt_c[i-1] * math.exp(math.log(opt[i-1] + 1)))
    # Calculate the quality of the portfolio
    n_eq, n_opt = numpy.array(eq), numpy.array(opt)
    eq_risk, opt_risk = float(numpy.array(eq).std()), float(numpy.array(opt).std())
    eq_return, opt_return = (eq_c[len(eq_c) - 1]/eq_c[0])-1, (opt_c[len(opt_c) - 1]/opt_c[0])-1
    print("Equal", " Risk =", eq_risk, " Return =", eq_return, " Sharpe =", float(n_eq.mean()) / eq_risk)
    print("Optimal", " Risk =", opt_risk, " Return =", opt_return, " Sharpe =", float(n_opt.mean()) / opt_risk)
    # Plot the compounded returns for each portfolio
    plt.style.use('bmh')
    plt.plot(eq_c, label="Equal Weights")
    plt.plot(opt_c, label="Optimal Weights")
    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    main()
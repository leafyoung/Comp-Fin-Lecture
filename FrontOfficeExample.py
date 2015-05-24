import os
import math
import numpy
import pandas
import random
import Quandl
import numpy.random as nrand
import matplotlib.pyplot as plt


class Swarm:
    def __init__(self, n, i, dim, c1, c2, w, p):
        self.iterations = i
        self.swarm = []
        self.p = p
        self.dim = dim
        self.c1, self.c2, self.w = c1, c2, w
        for i in range(n):
            self.swarm.append(Particle(dim, c1, c2, w, p))

    def optimize(self):
        fits = []
        best_g_best, best_g_best_fit = None, float('-inf')
        for i in range(self.iterations):
            g_best, f, avg_f = self.get_best()
            fits.append(avg_f)
            if f > best_g_best_fit:
                best_g_best = g_best
                best_g_best_fit = f
            for particle in self.swarm:
                if particle is not g_best:
                    particle.update(g_best.x)
                    particle.x = self.p.repair(particle.x)
                    distance = (particle.x - g_best.x)**2
                    if numpy.sum(distance) < 0.1:
                        particle.x = nrand.uniform(self.p.bounds[0], self.p.bounds[1], self.dim)
                        particle.x = self.p.repair(particle.x)
                else:
                    for j in range(25):
                        neighbour_x = nrand.uniform(self.p.bounds[0], self.p.bounds[1], self.dim)
                        neighbour_x = self.p.repair(neighbour_x)
                        if self.p.fit(neighbour_x) > f:
                            particle.x = neighbour_x
            if len(fits) > 250:
                change = (fits[i] / fits[i-50]) - 1
                if -0.001 <= change <= 0.001:
                    break
        return best_g_best.x, best_g_best_fit

    def get_best(self):
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
        self.dim, self.c1, self.c2, self.w, self.p = dim, c1, c2, w, p
        self.x = nrand.uniform(p.bounds[0], p.bounds[1], dim)
        self.v = nrand.uniform(0.000, 0.001, dim)
        self.p_best, self.p_best_fit = self.x, float('-inf')

    def update(self, g_best):
        c1r1 = nrand.uniform(0, 1, self.dim) * self.c1
        c2r2 = nrand.uniform(0, 1, self.dim) * self.c2
        cog = (self.x - self.p_best) * c1r1
        soc = (self.x - g_best) * c2r2
        self.v += cog + soc
        self.x += self.v * self.w
        if self.p.fit(self.x) > self.p_best_fit:
            self.p_best = self.x


class Portfolio:
    def __init__(self, bounds, rets, risk, corr):
        self.bounds = bounds
        self.er = rets
        self.es = risk
        self.c = corr

    def fit(self, x):
        r = numpy.sum(self.er * x)
        risk = numpy.sum(self.es**2 * x**2)
        for i in range(len(x)):
            for j in range(len(x)):
                risk += x[i] * x[j] * self.es[i] * self.es[j] * self.c.item((i, j))
        sharpe = r / math.sqrt(risk)
        return sharpe

    def repair(self, x):
        norm = x / numpy.sum(x)
        for i in range(len(norm)):
            norm[i] = max(norm[i], self.bounds[0])
            norm[i] = min(norm[i], self.bounds[1])
        norm_x = norm / numpy.sum(norm)
        return norm_x


class Fetcher:
    def __init__(self, quandl_auth):
        """
        :param quandl_auth:
        """
        self.token = quandl_auth

    def get_stock(self, ticker, start, end, drop=None, collapse="daily", transform="rdiff"):
        """
        :param ticker:
        :param start:
        :param end:
        :param drop:
        :param collapse:
        :param transform:
        :return:
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
        :param tickers:
        :param start:
        :param end:
        :param drop:
        :param collapse:
        :param transform:
        :return:
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
    tickers = list(pandas.read_csv("#PortfolioStocks.csv")["Ticker"])
    start_date, end_date = "2005-01-01", "2015-01-01"
    fetcher = Fetcher("N9HccV672zuuiU5MUvcq")
    data = fetcher.get_stocks(tickers, start_date, end_date)

    equal = numpy.empty(len(data.columns))
    equal.fill(float(1/len(data.columns)))
    optimized_weights = equal

    eq, opt = [], []
    start, ym, mm, dm = 0, "2010", "01", "01"
    for i in range(len(data)):
        dd = str(data.index[i]).split('-')
        y, m, d = dd[0], dd[1], dd[2]

        if m != mm and int(m) % 3 == 0:
            data_slice = data.iloc[start:i]
            one_plus = data_slice + 1

            risk = numpy.array(data_slice.std())
            corr = numpy.matrix(data_slice.corr())
            rets = numpy.array(one_plus.product())
            rets = numpy.log(rets)

            eq_ret = float(numpy.sum(rets * equal))
            opt_ret = float(numpy.sum(rets * optimized_weights))

            eq.append(eq_ret)
            opt.append(opt_ret)

            p = Portfolio([0.00, 0.25], rets, risk, corr)
            s = Swarm(50, 5000, len(data.columns), 1.496180, 1.496180, 0.729844, p)
            optimized_weights, f = s.optimize()

            print(d, m, y, eq_ret, opt_ret, f)
            ym, mm, dm = y, m, d
            start = i

    eq_c, opt_c = [1.0], [1.0]
    for i in range(1, len(eq)):
        eq_c.append(eq_c[i-1] * math.exp(math.log(eq[i-1] + 1)))
        opt_c.append(opt_c[i-1] * math.exp(math.log(opt[i-1] + 1)))

    eq_risk, opt_risk = float(numpy.array(eq).std()), float(numpy.array(opt).std())
    eq_return, opt_return = (eq_c[len(eq_c) - 1]/eq_c[0])-1, (opt_c[len(opt_c) - 1]/opt_c[0])-1
    print("Equal Weights", " Risk =", eq_risk, " Return =", eq_return, " Sharpe =", eq_return / eq_risk)
    print("Optimal Weights", " Risk =", opt_risk, " Return =", opt_return, " Sharpe =", opt_return / opt_risk)

    plt.plot(eq_c, label="Equal Weights")
    plt.plot(opt_c, label="Optimal Weights")

    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    main()
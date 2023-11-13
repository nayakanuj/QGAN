import matplotlib.pyplot as plt
import numpy as np


class plotter:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs

    def plot_dist(self, data_round, data_range, sim_support, sim_dist, num_dim=1):
        if num_dim == 1:

            plt.figure()
            sim_support = sim_support.flatten()
            num_bins = len(sim_dist)
            plt.plot(sim_support, sim_dist, 'b-o', label="simulation")
            plt.xticks(np.arange(min(sim_support), max(sim_support) + 1, 1.0))
            (h, w) = np.histogram(data_round, bins=np.arange(data_range[0] - 0.5, data_range[1] + 1.5, 1))
            plt.bar((w[0:-1] + w[1:]) / 2, h / len(data_round), color = 'cyan', width = 0.5, label="data")
            plt.grid()
            plt.xlabel("x")
            plt.ylabel("p(x)")
            plt.legend(loc="best")
            plt.show(block=False)

            plt.figure()
            # plt.bar((w[0:-1]+w[1:])/2, np.cumsum(h)/len(data_round), color='blue', width=0.2, label="data distribution")
            plt.title("CDF")
            plt.plot((w[0:-1] + w[1:]) / 2, np.cumsum(h) / len(data_round), 'b-o', label="data")
            plt.bar(sim_support, np.cumsum(sim_dist), color='red', width=0.2, label="simulation")
            plt.grid()
            plt.xlabel("x")
            plt.ylabel("P(x)")
            plt.legend(loc="best")
            plt.show(block=False)
        elif num_dim == 2:
            pass
        else:
            raise Exception("Number of dimensions >= 2 not supported")

        return

    def plot_rel_entropy(self, rel_entr):
        # Plot progress w.r.t relative entropy
        plt.figure(figsize=(6, 5))
        plt.title("Relative Entropy")
        plt.plot(np.linspace(0, self.num_epochs, len(rel_entr)), rel_entr, 'k-s')
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("$D_{KL}(p_{data} || p_{g})$")
        plt.show(block=False)

    def plot_loss(self, g_loss, d_loss):
        # Plot progress w.r.t the generator's and the discriminator's loss function
        t_steps = np.arange(self.num_epochs)
        plt.figure(figsize=(6, 5))
        plt.title("Loss function")
        plt.plot(t_steps, g_loss, "r-o", label="Generator", linewidth=2)
        plt.plot(t_steps, d_loss, "b-^", label="Discriminator", linewidth=2)
        plt.grid()
        plt.legend(loc="best")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show(block=False)

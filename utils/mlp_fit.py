
import numpy as np
import pylab as plt
import torch
from torchvision.ops.misc import MLP
from torch.utils.data import DataLoader


class CurveFitMLP:

    def __init__(self, indep_var, dep_var, num_layer=4, hsize=10, batch_size=None, lr=1e-3, mom=0.9):
        """

        Parameters
        ----------
        indep_var, np.ndarray of the independent variable (feature)
        dep_var, np.ndarray of the dependent variable (metric of interest)
        num_layer, number of hidden FC layers
        hsize, number of input/output channels for hidden layers
        batch_size, size of each batch for DataLoader
        lr, learning rate for SGD
        mom, momentum for SGD
        """
        self.x = indep_var
        self.y = dep_var
        self.lr = lr
        self.mom = mom
        self.hidden_dims = [hsize]*num_layer + [1]
        self.model = None
        self.optimizer = None
        self.loss = torch.nn.L1Loss()
        self.data_xy = torch.tensor(np.array([self.x, self.y]).T.astype(np.float32))
        if batch_size is None:
            batch_size = int(0.25 * len(self.x))
            print("batch size=%d" % batch_size)
        self.data_xy = DataLoader(self.data_xy, batch_size=batch_size, shuffle=True)

        self.initialize_model()

    def initialize_model(self):
        """initializes the MLP model and optimizer"""
        input_size = 1
        self.model = MLP(input_size, self.hidden_dims)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.mom)

    def train_model(self, n_ep):
        """
        trains the model for n_ep (int) epochs
        """
        self.model = self.model.train()
        num_batch = len(self.data_xy)
        for i_ep in range(n_ep):
            for i_batch, xy in enumerate(self.data_xy):
                inputs = xy[:, 0][:, None]
                labels = xy[:, 1][:, None]

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                error = self.loss(outputs, labels)
                error.backward()
                self.optimizer.step()
                print("Ep %d / %d, batch %d / %d, loss=%.4f" %
                        (i_ep+1, n_ep, i_batch+1, num_batch, error.item()))

    def get_yfit(self, x):
        """evaluates the current model at x, where x is a numpy array"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy array type")
        xtens = torch.tensor(x.astype(np.float32))[:, None]
        self.model = self.model.eval()
        with torch.no_grad():
            yfit = self.model(xtens)
        return yfit.numpy()

    def save_model(self, filename):
        torch.save({"mod": self.model.state_dict(), "hidden_dims": self.hidden_dims}, filename)

    @staticmethod
    def load_model(filename):
        model_dict = torch.load(filename)
        hidden_dims = model_dict["hidden_dims"]
        model = MLP(1, hidden_dims)
        model.load_state_dict(model_dict["mod"])
        model.eval()
        return model


if __name__ == "__main__":
    # Example of using the above class
    # inspired from:
    # https://michael-franke.github.io/npNLG/04-ANNs/04d-MLP-pytorch.html
    def target(x):
        return x**3 - x**2 + 25 * np.sin(2*x)

    x = np.linspace(start=-5, stop=5, num=1000)
    y = target(x)

    n_obs = 100  # number of observations
    xobs = np.random.uniform(x.min() + 1e-6, x.max() - 1e-6, n_obs)
    yobs = np.random.normal(target(xobs), 10)

    plt.plot(x, y, label="truth", lw=2)
    # plot the data
    plt.plot(xobs, yobs, 'o', label="obs")
    # plot the function
    curve_fit = CurveFitMLP(xobs, yobs, lr=1e-3)
    curve_fit.train_model(n_ep=2000)
    yfit = curve_fit.get_yfit(x)
    plt.plot(x, yfit, ls='--', lw=2, label="MLP fit")

    # train again with different lr
    curve_fit.lr=1e-4
    curve_fit.initialize_model()
    curve_fit.train_model(n_ep=3000)
    yfit = curve_fit.get_yfit(x)
    plt.plot(x, yfit, ls='--', lw=2, label="MLP fit")

    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.legend(prop={"size": 12})
    plt.gca().grid(1, ls="--", lw=0.5)
    plt.show()

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from torch import nn
import pymc
seed = 1

def train_linear_model(X,y):
    model = LinearRegression()
    model.fit(X,y)
    return model

def read_data(filename):
    data = torch.tensor([list(map(float, line.split())) for line in open(filename)], dtype=torch.float32)
    return data[:, 0], data[:, 1]

'''
Code adapted from 
https://pytorch.org/tutorials/beginner/basics/quickstart\_tutorial.html


'''


# code task 11
# main neural network function
def find_nn():

    X_train, y_train = read_data('regression_train.txt')
    X_test, y_test = read_data('regression_test.txt')
    # normalising the data
    X_train_mean, X_train_std = X_train.mean(), X_train.std()
    X_train_norm = (X_train - X_train_mean) / X_train_std
    X_test_norm = (X_test - X_train_mean) / X_train_std
    y_train_min, y_train_max = y_train.min(), y_train.max()
    y_train_norm = (y_train - y_train_min) / (y_train_max - y_train_min)
    y_test_norm = (y_test - y_train_min) / (y_train_max - y_train_min)

    X_train_norm = X_train_norm.view(-1, 1)
    y_train_norm = y_train_norm.view(-1, 1)

    X_test_norm = X_test_norm.view(-1, 1)
    y_test_norm = y_test_norm.view(-1, 1)

    ## allocating the evaluation data as the rightmost section

    train_data = torch.cat((X_train_norm, y_train_norm), dim=1)
    sorted_data = train_data[train_data[:, 0].argsort()]
    train_data_final = sorted_data
    X_train_norm = train_data_final[:, 0].view(-1, 1)
    y_train_norm = train_data_final[:, 1].view(-1, 1)

    val_x = train_data_final[-30:, 0].view(-1, 1)
    val_y = train_data_final[-30:, 1].view(-1, 1)

    torch.manual_seed(1)
    model = NeuralNetworkModel()
    # MEAN ERROR LOSS
    loss_fn = nn.MSELoss()
    learning_rate = 0.001
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 1862
    stationary = 0
    min_val_loss = float('inf')
    patience = 200

    for epoch in range(epochs):

        model.train()
        y_pred = model(X_train_norm)
        loss = loss_fn(y_pred, y_train_norm)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (epoch + 1) % 1 == 0: # evaluate the model at every epoch
            model.eval()

            with torch.no_grad():
                val_pred = model(val_x)
                val_loss = loss_fn(val_pred, val_y)
            # print("validation loss", val_loss)
            # checking for decreasing validation loss (does so withing certain decimals so insignificant change is ignored)
            if val_loss.item() < min_val_loss:
                min_val_loss = val_loss.item()
                stationary = 0
            else:
                stationary += 1
                print("NO IMPROVEMENT", stationary, "/",patience)
            # early stopping
            if stationary >= patience:
                print("STOPPING")
                break
                
            print("EPOCH: ", epoch, "VAL LOSS:", val_loss.item())

        # PLOTTING
    x_train_range_norm = torch.linspace(X_train_norm.min(), X_train_norm.max(), steps=500).view(-1, 1)
    x_test_range_norm = torch.linspace(X_test_norm.min(), X_test_norm.max(), steps=500).view(-1, 1)

    model.eval()
    with torch.no_grad():
        y_train_range_pred = model(x_train_range_norm)
    with torch.no_grad():
        y_test_range_pred = model(x_test_range_norm)

    X_train = X_train.numpy()
    y_train_np = y_train.numpy()
    X_test_np = X_test.numpy()
    y_test_np = y_test.numpy()

    x_train_range_np = (x_train_range_norm * X_train_std + X_train_mean).numpy().flatten()
    x_test_range_np = (x_test_range_norm * X_train_std + X_train_mean).numpy().flatten()
    y_train_range_np = denormalise(y_train_range_pred, y_train_min, y_train_max).numpy().flatten()
    y_test_range_np = denormalise(y_test_range_pred, y_train_min, y_train_max).numpy().flatten()

    plt.figure(figsize=(10, 6))
    plt.plot(x_train_range_np, y_train_range_np, label="Fitted Model", color=(0, 0, 1), linewidth=2)
    plt.scatter(X_train, y_train, label="Training Data", color=(0, 1, 0), s=50)
    plt.title("Neural Network vs data")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x_test_range_np, y_test_range_np, label="Fitted Model", color=(0, 0, 1), linewidth=2)
    plt.scatter(X_test, y_test, label="Testing Data", color=(1, 0, 0), s=50)
    plt.title("Neural Network vs data")
    plt.legend()
    plt.grid(True)
    plt.show()


    model.eval()
    with torch.no_grad():
        # predictions for train validation test sets
        train_pred = model(X_train_norm)
        test_pred = model(X_test_norm)
        val_x_norm = (val_x - X_train_mean) / X_train_std
        val_y_norm = (val_y - y_train_min) / (y_train_max - y_train_min)

        val_pred = model(val_x_norm)

        train_pred_rescaled = denormalise(train_pred, y_train_min, y_train_max)
        test_pred_rescaled = denormalise(test_pred, y_train_min, y_train_max)
        val_pred_rescaled = denormalise(val_pred, y_train_min, y_train_max)
        y_train = y_train.view(-1, 1)
        y_test = y_test.view(-1, 1)
        # y_val = val_x.view(-1, 1)

        train_mse = loss_fn(train_pred_rescaled, y_train)
        test_mse = loss_fn(test_pred_rescaled, y_test)
        val_mse = loss_fn(val_pred_rescaled, val_y_norm)

        print("Final Losses:")
        print("Train MSE:", train_mse.item())
        print("Val MSE:", val_mse.item())
        print("test MSE:",test_mse.item())
    '''
    Early stopping implemented. Patience = 200


    Two layers: 424 epochs
    Train MSE: 4327.746094
    Test MSE: 1641043.875000


    Single Layer:With ReLU:(1121 Epochs):
    Train MSE: 3738.287842
    Test MSE: 6593.163086


    Without ReLU: 2061 epochs
    Final Losses:
    Train MSE: 3728.574463
    Test MSE: 4645.291992

    Since that has the best performance, a single layer is most suitable for this problem
    From this, subtract 200 from 2061 epochs, as it is known to converge (reduce overfitting)

    Updated values:
    Final Losses:
    Train MSE: 3764.12744140625
    Val MSE: 2421.91748046875
    test MSE: 4317.03759765625
    
        
    (seed-specific)


    '''

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, output_size=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, output_size),
            # # nn.ReLU(),
            # nn.Linear(64, 128),
            # # nn.ReLU(),
            # nn.Linear(128, 64),
            # # nn.ReLU(),
            # nn.Linear(64, 1),
            # # nn.ReLU(),

        )

    def forward(self, x):
        c = torch.ones_like(x)
        x_poly = torch.cat([
            c,  # constant
            x,  # lin
            x ** 2,  #  quad
            x ** 3], dim=1) # cubic
            # x ** 4, # tested up to x^7 all had lower validaiton loss- so only using up to x^3


        return self.network(x_poly)


def denormalise(y_norm, y_min, y_max):
    return y_norm * (y_max - y_min) + y_min



# task 10
def find_best_poly():
    # loading the data
    train = np.loadtxt("regression_train.txt")
    X_train = train[:, 0].reshape(-1, 1)
    y_train = train[:, 1]

    test = np.loadtxt("regression_test.txt")
    X_test = test[:, 0].reshape(-1, 1)
    y_test = test[:, 1]


    for i in range(1, 4):  # for varying polynomial degrees
        poly = PolynomialFeatures(degree=i, include_bias=False)

        val_size = 35
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train_scaled = poly.fit_transform(X_train)
        X_test_scaled = poly.transform(X_test)
        X_val_scaled = poly.transform(X_val)
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred_val = model.predict(X_val_scaled)
        y_pred_test = model.predict(X_test_scaled)
        y_pred_train = model.predict(X_train_scaled)


        mse_train = mean_squared_error(y_train, y_pred_train)
        print("Poly degree", i, "- Train MSE: ", mse_train)
        mse_val = mean_squared_error(y_val, y_pred_val)
        print("Poly degree", i, "- Validation MSE: ", mse_val)
        mse_test = mean_squared_error(y_test, y_pred_test)
        print("Poly degree", i, "- Test MSE: ", mse_test)


# task 12
def train_bayes(x, y):
    with pymc.Model() as model:
        # we know that the data follows a cubic

        # uniform noise as given in task description
        noise = pymc.Uniform("sigma", lower=0, upper=200)
        intercept = pymc.Normal("intercept", mu=0, sigma=20)
        m1 = pymc.Normal("m1", mu=0, sigma=20)
        m2 = pymc.Normal("m2", mu=0, sigma=20)
        m3 = pymc.Normal("m3", mu=0, sigma=20)

        mu = intercept + m1 * x + m2 * x**2 + m3 * x**3 # we already know it's a cubic function
        # calculate likelihood with uniform noise
        likelihood = pymc.Normal("y", mu=mu, sigma=noise, observed=y)
        # posterior
        trace = pymc.sample(1000, cores=8)
    # getting the values
    mean_intercept = trace.posterior["intercept"].mean().values
    mean_m1 = trace.posterior["m1"].mean().values
    mean_m2 = trace.posterior["m2"].mean().values
    mean_m3 = trace.posterior["m3"].mean().values
    mean_sigma = trace.posterior["sigma"].mean().values
    print(mean_intercept, mean_m1, mean_m2, mean_m3, mean_sigma)
    return mean_intercept, mean_m1, mean_m2, mean_m3, mean_sigma



def main():

    # linear regression
    find_best_poly()
    # # neural network
    find_nn()

    # bayesian regression
    data = np.loadtxt("regression_train.txt")
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    # run 5 chains
    total_intercept = 0
    total_m1 = 0
    total_m2 = 0
    total_m3 = 0
    total_sigma = 0
    chains = 5
    for i in range(chains):
        print(X.shape, y.shape)
        intercept, m1, m2, m3, sigma = train_bayes(X.reshape(-1), y)
        total_intercept += intercept
        total_m1 += m1
        total_m2 += m2
        total_m3 += m3
        total_sigma += sigma

    print(total_intercept/chains, total_m1/chains, total_m2/chains, total_m3/chains, total_sigma/chains)

if __name__ == "__main__":
    main()
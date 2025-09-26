import numpy as np
import matplotlib.pyplot as plt

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
        x (ndarray (m,)): Data, m examples
        w, b (scalar)   : model parameters
    Returns
        f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb

# Training set has 3 data points.
x_train = np.array([1.0, 2.0, 3.0])
y_train = np.array([300.0, 400.0, 700.0])
print(f"x_train = {x_train}")                       # x_train = [1. 2. 3.]
print(f"y_train = {y_train}")                       # y_train = [300. 400. 700.]

# Number of training examples is 3.
print(f"x_train.shape: {x_train.shape}")            # x_train.shape: (3,)
m = x_train.shape[0]
print(f"Number of training examples is: {m}")       # Number of training examples is: 3

w = 186
b = 100
tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot the data
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
plt.xlim(left=0)
plt.ylim(bottom=0)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')

plt.title("Housing Prices")                         # Set the title
plt.ylabel("Price (in 1000s of dollars)")           # Set the y-axis label
plt.xlabel("Size (1000 sqft)")                      # Set the x-axis label
plt.legend()
plt.show()
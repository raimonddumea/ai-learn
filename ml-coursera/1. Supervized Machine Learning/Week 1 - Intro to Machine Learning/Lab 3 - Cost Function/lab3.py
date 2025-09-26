import numpy as np
import matplotlib.pyplot as plt

def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    m = x.shape[0]
    cost_sum = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost

    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost

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
tmp_f_wb = compute_cost(x_train, y_train, w, b)

# Plot the cost function
# Create a range of w values to plot
w_range = np.linspace(0, 400, 100)
b_fixed = 100  # Keep b fixed

# Calculate cost for each w value
cost_values = []
for w_val in w_range:
    cost = compute_cost(x_train, y_train, w_val, b_fixed)
    cost_values.append(cost)

# Plot the cost curve
plt.plot(w_range, cost_values, 'b-', linewidth=2)

# Add a marker at the current w value
cost_at_w = compute_cost(x_train, y_train, w, b_fixed)
plt.plot(w, cost_at_w, 'ro', markersize=10, label=f'cost at w={w}')

# Set labels and title
plt.title(f"Cost vs. w, (b fixed at {b_fixed})")
plt.xlabel('w')
plt.ylabel('Cost')
plt.legend()
plt.show()
import numpy as np

def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g

# Input is a single number
input_val = 1
exp_val = np.exp(input_val)
print("Input to exp: ", input_val)      # 1
print("Output of exp: ", exp_val)       # 2.718281828459045

# Input is an array
input_array = np.array([1, 2, 3])
exp_array = np.exp(input_array)
print("Input to exp: ", input_array)    # [1 2 3]
print("Output of exp: ", exp_array)     # [ 2.71828183  7.3890561  20.08553692]

# Generate an array of evenly spaced values between -10 and 10
z_tmp = np.arange(-10, 11)
print(z_tmp)                            # [ -10   -9   -8   -7   -6   -5   -4   -3   -2   -1    0    1    2    3    4    5    6    7    8    9   10]

# Get the sigmoid values
y = sigmoid(z_tmp)

# Using list comprehension and join to format the output
formatted_values = ' '.join([f"{val:.2f}" for val in y])
print(f"[{formatted_values}]")          # [0.00 0.00 0.00 0.00 0.00 0.01 0.02 0.05 0.12 0.27 0.50 0.73 0.88 0.95 0.98 0.99 1.00 1.00 1.00 1.00 1.00]                       


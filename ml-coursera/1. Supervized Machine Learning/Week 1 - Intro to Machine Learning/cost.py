import numpy

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost

    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost

# Training set has 3 data points.
x_train = numpy.array([1.0, 2.0, 3.0])
y_train = numpy.array([300.0, 400.0, 700.0])

w = 200
b = 100

tmp_f_wb = compute_cost(x_train, y_train, w, b)
print(tmp_f_wb)
import numpy

def compute_gradient(x, y, w, b): 
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    b = b_in
    w = w_in

    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w , b)

        b = b - alpha * dj_db
        w = w - alpha * dj_dw

    return w, b

# Training set has 3 data points.
x_train = numpy.array([1.0, 2.0, 3.0])
y_train = numpy.array([300.0, 400.0, 700.0])

# initialize parameters
w_init = 0
b_init = 0

# some gradient descent settings
iterations = 10000
learning_rate = 1.0e-2

# run gradient descent
w_final, b_final = gradient_descent(x_train ,y_train, w_init, b_init, learning_rate, iterations)
print(f"(w,b) found by gradient descent: ({w_final:.2f},{b_final:.2f})")
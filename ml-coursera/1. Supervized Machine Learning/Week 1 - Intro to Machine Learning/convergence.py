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

def gradient_descent(x, y, w_in, b_in, max_iters, alpha, tol):
    w = w_in
    b = b_in
    iteration = 0

    while iteration < max_iters:
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # Store old values to check convergence
        w_old = w
        b_old = b

        # Update parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        iteration += 1

        # Check if converged (parameters barely changing)
        if abs(w - w_old) < tol and abs(b - b_old) < tol:
            print(f"Converged at iteration {iteration}")
            break
    else:
        print(f"Warning: Reached max iterations ({max_iters}) without converging")

    return w, b

# Training set has 3 data points.
x_train = numpy.array([1.0, 2.0, 3.0])
y_train = numpy.array([300.0, 400.0, 700.0])

# initialize parameters
w_init = 0
b_init = 0

# some gradient descent settings
max_iterations = 10000
learning_rate = 1.0e-2
tolerance = 1.0e-6

# run gradient descent
w_final, b_final = gradient_descent(x_train ,y_train, w_init, b_init, max_iterations, learning_rate, tolerance)
print(f"(w,b) found by gradient descent: ({w_final:.2f},{b_final:.2f})")
import numpy as np
import matplotlib.pyplot as plt

def plot_lines(M, b=None, x_range=(-10, 10)):
    """
    Plot lines representing a system of linear equations.

    Parameters:
    M: Either a 2D array of coefficients or an augmented matrix [A|b]
    b: 1D array of constants (optional if M is augmented)
    x_range: tuple of (min, max) for x-axis range
    """
    # If b is not provided, assume M is an augmented matrix
    if b is None:
        # Split augmented matrix into coefficients and constants
        coefficients = M[:, :-1]
        constants = M[:, -1]
    else:
        coefficients = M
        constants = b

    x = np.linspace(x_range[0], x_range[1], 100)

    fig, ax = plt.subplots()

    for i in range(len(coefficients)):
        if coefficients[i][1] != 0:  # Avoid division by zero
            # Solve for y: a*x + b*y = c -> y = (c - a*x) / b
            y = (constants[i] - coefficients[i][0] * x) / coefficients[i][1]
            ax.plot(x, y, label=f'Line {i+1}: {coefficients[i][0]}x + {coefficients[i][1]}y = {constants[i]}')
        else:
            # Vertical line: x = c/a
            if coefficients[i][0] != 0:
                x_val = constants[i] / coefficients[i][0]
                ax.axvline(x=x_val, label=f'Line {i+1}: x = {x_val}')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)
    ax.set_title('System of Linear Equations')
    plt.show()
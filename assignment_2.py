import numpy as np

#Number 1 ////Nevviles method
def neville_interpolation(x_vals, y_vals, x_target):
    """
    Uses Neville's method to interpolate the function at a given x_target.

    Parameters:
    x_vals (list): List of x values.
    y_vals (list): Corresponding list of function values f(x).
    x_target (float): The x value to interpolate.

    Returns:
    float: The interpolated function value at x_target.
    """
    n = len(x_vals)
    Q = np.zeros((n, n))
    Q[:, 0] = y_vals  #  given function values

    for j in range(1, n):
        for i in range(n - j):
            Q[i, j] = ((x_target - x_vals[i + j]) * Q[i, j - 1] - (x_target - x_vals[i]) * Q[i + 1, j - 1]) / (x_vals[i] - x_vals[i + j])

    return Q[0, -1]  # final interpolated value us the  top right one in the outpit 

# Gata points from the question
x_neville = [3.6, 3.8, 3.9]
y_neville = [1.675, 1.436, 1.318]
x_target = 3.7

# Compute
neville_result = neville_interpolation(x_neville, y_neville, x_target)
# Display result
neville_result


#number 2
def newton_forward_diff(x_vals, y_vals):
    """
    Constructs Newton's Forward Difference Table.
#list all parameters 
    Parameters:
    x_vals (list): List of x values.
    y_vals (list): Corresponding list of function values f(x).

    Returns:
    numpy.ndarray: Forward difference table.
    """
    n = len(x_vals)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y_vals  # this is from the First column which is is the given function values

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = diff_table[i + 1, j - 1] - diff_table[i, j - 1]

    return diff_table

# Given data points
x_newton = [7.2, 7.4, 7.5, 7.6]
y_newton = [23.5492, 25.3913, 26.8224, 27.4589]

# Compute the forward difference table
diff_table = newton_forward_diff(x_newton, y_newton)

# Display result
print("Newton Forward Difference Table:")
print(diff_table)



#number 3
def newton_interpolation(x_vals, y_vals, x_target):
    """
    Uses Newton's Forward Difference method to approximate f(x_target) using only NumPy.
#list the parameters so be claer 
    Parameters:
    x_vals (numpy array): List of x values.
    y_vals (numpy array): Corresponding list of function values f(x).
    x_target (float): The x value to approximate.

    Returns:
    float: The interpolated function value at x_target.
    """
    n = len(x_vals)
    diff_table = newton_forward_diff(x_vals, y_vals)
    h = x_vals[1] - x_vals[0]  # Step size
    s = (x_target - x_vals[0]) / h  # Scaling factor
    interpolation = y_vals[0]
    term = 1

    for i in range(1, n):
        term *= (s - (i - 1)) / i  # this is the factorial term
        interpolation += term * diff_table[0, i]

    return interpolation

# target x valu
x_newton = np.array([7.2, 7.4, 7.5, 7.6])
y_newton = np.array([23.5492, 25.3913, 26.8224, 27.4589])
x_approx = 7.3

# Compute f(7.3)
newton_result = newton_interpolation(x_newton, y_newton, x_approx)

# Print result
print("Approximated value of f(7.3) is")
print(newton_result)


#number 4

def hermite_interpolation(x_vals, y_vals, dy_vals):
    """
    Constructs the Hermite interpolation divided difference table.

    Parameters:
    x_vals (list): List of x values.
    y_vals (list): Corresponding list of function values f(x).
    dy_vals (list): Corresponding list of derivative values f'(x).

    Returns:
    numpy.ndarray: Hermite divided difference table.
    """
    n = len(x_vals)
    z = np.zeros(2 * n)  # Expand the x-values to include duplicates
    Q = np.zeros((2 * n, 2 * n))  # this is what were looking for in number 5 

    # Fill in table
    for i in range(n):
        z[2 * i] = x_vals[i]
        z[2 * i + 1] = x_vals[i]
        Q[2 * i, 0] = y_vals[i]
        Q[2 * i + 1, 0] = y_vals[i]
        Q[2 * i + 1, 1] = dy_vals[i]

        if i > 0:
            Q[2 * i, 1] = (Q[2 * i, 0] - Q[2 * i - 1, 0]) / (z[2 * i] - z[2 * i - 1])

    # Compute divided differences
    for j in range(2, 2 * n):
        for i in range(2 * n - j):
            Q[i, j] = (Q[i + 1, j - 1] - Q[i, j - 1]) / (z[i + j] - z[i])

    return Q

# data points
x_hermite = [3.6, 3.8, 3.9]
y_hermite = [1.675, 1.436, 1.318]
dy_hermite = [-1.195, -1.188, -1.182]

# find the the Hermite divided difference table
hermite_result = hermite_interpolation(x_hermite, y_hermite, dy_hermite)

# result
print("Hermite Divided Difference Table:")
print(hermite_result)





#number 5

# Re-import necessary libraries after execution reset
import numpy as np
import pandas as pd


def cubic_spline_matrix(x_vals, y_vals):
    """
    Constructs the coefficient matrix A and vector b for cubic spline interpolation.

    Parameters:
    x_vals (list): List of x values.
    y_vals (list): Corresponding list of function values f(x).

    Returns:
    tuple: (A matrix, b vector) for cubic spline calculation.
    """
    n = len(x_vals) - 1
    h = np.diff(x_vals)  # Step sizes between x values
    A = np.zeros((n - 1, n - 1))
    b = np.zeros(n - 1)

    for i in range(1, n):
        A[i - 1, i - 1] = 2 * (h[i - 1] + h[i])  # Diagonal elements
        if i != 1:
            A[i - 1, i - 2] = h[i - 1]  # Sub-diagonal elements
        if i != n - 1:
            A[i - 1, i] = h[i]  # Super-diagonal elements

        b[i - 1] = (3 / h[i]) * (y_vals[i + 1] - y_vals[i]) - (3 / h[i - 1]) * (y_vals[i] - y_vals[i - 1])

    return A, b

# Given data points
x_spline = [2, 5, 8, 10]
y_spline = [3, 5, 7, 9]

# Compute matrix A and vector b
A_matrix, b_vector = cubic_spline_matrix(x_spline, y_spline)

# Solve for vector x (coefficients for second derivatives)
x_vector = np.linalg.solve(A_matrix, b_vector)

# Display results
print("Matrix A:")
print(A_matrix)

print("\nVector b:")
print(b_vector)

print("\nVector x:")
print(x_vector)

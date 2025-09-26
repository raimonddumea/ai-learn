import numpy as np

# ===============================================
# 1. Basics of NumPy
# ===============================================

# Create and print a NumPy array 'a' containing the elements 1, 2, 3.
a = np.array([1, 2, 3])
print(a)                                        # [1 2 3]

# Create an array with 3 integers, starting from the default integer 0.
b = np.arange(3)
print(b)                                        # [0 1 2]

# Create an array that starts from the integer 1, ends before 20, incremented by 3.
c = np.arange(1, 20, 3)
print(c)                                        # [ 1  4  7 10 13 16 19]

# Create an array with five evenly spaced values in the interval from 0 to 100.
lin_spaced_arr = np.linspace(0, 100, 5)
print(lin_spaced_arr)                           # [  0.  25.  50.  75. 100.]
# Notice default type is np.float64.

# Create an array with five evenly spaced integer values in the interval from 0 to 100.
lin_spaced_arr = np.linspace(0, 100, 5, dtype=int)
print(lin_spaced_arr)                           # [  0  25  50  75 100]

# Create an array with a string and print the data type of the array.
char_arr = np.array(['Welcome to Math for ML!'])
print(char_arr.dtype)                           # <U23
# '<U23' means 23-character unicode string (U) on a little-endian architecture (<)

# Return a new array of shape 3, filled with ones. 
ones_arr = np.ones(3)
print(ones_arr)                                 # [1. 1. 1.]

# Return a new array of shape 3, filled with zeroes.
zeros_arr = np.zeros(3)
print(zeros_arr)                                # [0. 0. 0.]

# Return a new array of shape 3, without initializing entries.
empt_arr = np.empty(3)
print(empt_arr)                                 # [? ? ?]
# Contains whatever garbage values were in memory.
# Faster since no initialization.
# Values are unpredictable and should be overwritten before use.

# Return a new array of shape 3 with random numbers between 0 and 1.
rand_arr = np.random.rand(3)
print(rand_arr)                                 # [0.29635323 0.1618534  0.11794943]

# ===============================================
# 2. Multidimensional Arrays
# ===============================================

# Create a 2 dimensional array (2-D)
two_dim_arr = np.array([[1,2,3], [4,5,6]])
print(two_dim_arr)                              # [[1 2 3][4 5 6]]

# An alternative way to create a multidimensional array 
# is by reshaping the initial 1-D array.

# 1-D array 
one_dim_arr = np.array([1, 2, 3, 4, 5, 6])

# Multidimensional array using reshape()
multi_dim_arr = np.reshape(
                one_dim_arr,                    # the array to be reshaped
                (2,3)                           # dimensions of the new array
                )
# Print the new 2-D array with two rows and three columns
print(multi_dim_arr)                            # [[1 2 3][4 5 6]]

# Dimension of the 2-D array multi_dim_arr
print(multi_dim_arr.ndim)                       # 2

# Shape of the 2-D array multi_dim_arr
print(multi_dim_arr.shape)                      # (2, 3)

# Size of the array multi_dim_arr
print(multi_dim_arr.size)                       # 6

# ===============================================
# 3. Array math operations
# ===============================================

arr_1 = np.array([2, 4, 6])
arr_2 = np.array([1, 3, 5])

# Adding two 1-D arrays
addition = arr_1 + arr_2
print(addition)                                 # [ 3  7 11]

# Subtracting two 1-D arrays
subtraction = arr_1 - arr_2
print(subtraction)                              # [1 1 1]

# Multiplying two 1-D arrays elementwise
multiplication = arr_1 * arr_2
print(multiplication)                           # [ 2 12 30]

# Multiplying vector with scalar (broadcasting)
vector = np.array([1, 2])
km = vector * 1.6
print(km)                                       # [1.6 3.2]

# ===============================================
# 4. Indexing and Slicing
# ===============================================

# Select the third element of the array.
a = ([1, 2, 3, 4, 5])
print(a[2])                                     # 3

# Select the first element of the array.
print(a[0])                                     # 1

# Indexing on a 2-D array.
two_dim = np.array([[1, 2, 3],
                    [4, 5, 6], 
                    [7, 8, 9]])

# Select element number 8 from the 2-D array using indices i, j.
# i = 2 (3rd vector), j = 1 (2nd element)
print(two_dim[2][1])                            # 8

# Indexing on a four dimmensional array.
four_dim = np.array([
                     [
                      [
                       [1, 2, 3],
                       [4, 5, 6]
                      ],
                      [
                       [7, 8, 9],
                       [10, 11, 12]
                      ]
                     ],
                     [
                      [
                       [13, 14, 15],
                       [16, 17, 18]
                      ],
                      [
                       [19, 20, 21],
                       [22, 23, 24]
                      ]
                     ]
                    ])
                       
print(four_dim.ndim)                            # 4

# 2nd 3-D array, 1st 2-D array, 2nd vector, 3rd element)
print(four_dim[1][0][1][2])                     # 18

# Slice the array a to get the 3rd and 4th elements
a = ([1, 2, 3, 4, 5, 6, 7, 8])
sliced_arr = a[2:4]
print(sliced_arr)                               # [3, 4]

# Slice the array a to get the first 3 elements
sliced_arr = a[:3]
print(sliced_arr)                               # [1, 2, 3]

# Slice the array a to get all the elements starting with index 2 (the third one)
sliced_arr = a[2:]
print(sliced_arr)                               # [3, 4, 5, 6, 7, 8]

# Slice the array with a step of 2. Start and Stop parameters are empty.
sliced_arr = a[::2]
print(sliced_arr)                               # [1, 3, 5, 7]

# Note that a == a[:] == a[::]
print(a == a[:] == a[::])

two_dim = np.array([[1, 2, 3],
                    [4, 5, 6], 
                    [7, 8, 9]])

# Slice the two_dim array to get the first two rows.
sliced_arr_1 = two_dim[0:2]
print(sliced_arr_1)                             # [[1 2 3][4 5 6]]

# Similarily, slice the two_dim array to get the last two rows.
sliced_two_dim_rows = two_dim[1:3]
print(sliced_two_dim_rows)                      # [[4 5 6][7 8 9]]

# From all rows take the element at index 1 (the 2nd column).
sliced_two_dim_cols = two_dim[:,1]
print(sliced_two_dim_cols)                      # [2 5 8]

# ===============================================
# 5. Stacking
# ===============================================

a1 = np.array([[1,1], 
               [2,2]])
a2 = np.array([[3,3],
               [4,4]])

# Stack the arrays vertically
vert_stack = np.vstack((a1, a2))
print(vert_stack)                               # [[1 1][2 2][3 3][4 4]]

# Stack the arrays horizontally
horz_stack = np.hstack((a1, a2))
print(horz_stack)                               # [[1 1 3 3][2 2 4 4]]

two_dim = np.array([[1, 2, 3],
                    [4, 5, 6], 
                    [7, 8, 9]])

vertically_splitted = np.vsplit(two_dim, 3)
print(vertically_splitted[0])                   # [[1 2 3]]
print(vertically_splitted[1])                   # [[4 5 6]]
print(vertically_splitted[2])                   # [[7 8 9]]

horizontally_splitted = np.hsplit(two_dim, 3)
print(horizontally_splitted[0])                 # [[1][4][7]]
print(horizontally_splitted[1])                 # [[2][5][8]]
print(horizontally_splitted[2])                 # [[3][6][9]]

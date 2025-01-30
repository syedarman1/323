# Analysis of Algorithms (CSCI 323)
# Winter 2025
# Assignment 3 - Matrix Multiplication

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
import math
import Assignment02 as as2

#[2] Define a function random_matrix(mn, mx, rows, cols) that  returns a matrix of random integers in the range(mn, mx) inclusive. You can code a nested loop, or use list-comprehension like
def random_matrix(mn, mx, rows, cols):
    return [[random.randint(mn, mx) for _ in range(cols)] for _ in range(rows)]

#numpy_mult (wrap the function provided by Python's famous NumPy package)
def numpy_mult(mat1, mat2, n):
    return np.matmul(mat1, mat2)


#listcomp_mult (use Python's nested list comprehension)
def listcomp_mult(mat1, mat2, n):
    return [[sum([mat1[i][k] * mat2[k][j] for k in range(n)]) for j in range(n) ] for i in range(n) ]


#simple_mult (our first approach, typically taught in Discrete Math and Linear Algebra)
def simple_mult(mat1, mat2, n):
    res = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                res[i][j] += mat1[i][k] * mat2[k][j]

    return res


#divconq_mult (our second approach, basic divide-and-conquer algorithm)
# Function to add two matrices
def add_matrix(matrix_A, matrix_B, matrix_C, split_index):
    for i in range(split_index):
        for j in range(split_index):
            matrix_C[i][j] = matrix_A[i][j] + matrix_B[i][j]

        # Function to initialize matrix with zeros


# #def initWithZeros(a, r, c):
#     for i in range(r):
#         for j in range(c):
#             a[i][j] = 0


# Function to multiply two matrices
def multiply_matrix(matrix_A, matrix_B):
    col_1 = len(matrix_A[0])
    row_1 = len(matrix_A)
    col_2 = len(matrix_B[0])
    row_2 = len(matrix_B)

    if col_1 != row_2:
        print("\nError: The number of columns in Matrix A  must be equal to the number of rows in Matrix B\n")
        return 0

    result_matrix_row = [0] * col_2
    result_matrix = [[0 for x in range(col_2)] for y in range(row_1)]

    if col_1 == 1:
        result_matrix[0][0] = matrix_A[0][0] * matrix_B[0][0]

    else:
        split_index = col_1 // 2

        row_vector = [0] * split_index
        result_matrix_00 = [[0 for x in range(split_index)] for y in range(split_index)]
        result_matrix_01 = [[0 for x in range(split_index)] for y in range(split_index)]
        result_matrix_10 = [[0 for x in range(split_index)] for y in range(split_index)]
        result_matrix_11 = [[0 for x in range(split_index)] for y in range(split_index)]
        a00 = [[0 for x in range(split_index)] for y in range(split_index)]
        a01 = [[0 for x in range(split_index)] for y in range(split_index)]
        a10 = [[0 for x in range(split_index)] for y in range(split_index)]
        a11 = [[0 for x in range(split_index)] for y in range(split_index)]
        b00 = [[0 for x in range(split_index)] for y in range(split_index)]
        b01 = [[0 for x in range(split_index)] for y in range(split_index)]
        b10 = [[0 for x in range(split_index)] for y in range(split_index)]
        b11 = [[0 for x in range(split_index)] for y in range(split_index)]

        for i in range(split_index):
            for j in range(split_index):
                a00[i][j] = matrix_A[i][j]
                a01[i][j] = matrix_A[i][j + split_index]
                a10[i][j] = matrix_A[split_index + i][j]
                a11[i][j] = matrix_A[i + split_index][j + split_index]
                b00[i][j] = matrix_B[i][j]
                b01[i][j] = matrix_B[i][j + split_index]
                b10[i][j] = matrix_B[split_index + i][j]
                b11[i][j] = matrix_B[i + split_index][j + split_index]

        add_matrix(multiply_matrix(a00, b00), multiply_matrix(a01, b10), result_matrix_00, split_index)
        add_matrix(multiply_matrix(a00, b01), multiply_matrix(a01, b11), result_matrix_01, split_index)
        add_matrix(multiply_matrix(a10, b00), multiply_matrix(a11, b10), result_matrix_10, split_index)
        add_matrix(multiply_matrix(a10, b01), multiply_matrix(a11, b11), result_matrix_11, split_index)

        for i in range(split_index):
            for j in range(split_index):
                result_matrix[i][j] = result_matrix_00[i][j]
                result_matrix[i][j + split_index] = result_matrix_01[i][j]
                result_matrix[split_index + i][j] = result_matrix_10[i][j]
                result_matrix[i + split_index][j + split_index] = result_matrix_11[i][j]

    return result_matrix


def divide_conq_mult(mat1, mat2, n):
    return multiply_matrix(mat1, mat2)

#strassen_mult (our third approach, Strassen's creative/improved divide-and-conquer algorithm)
def split(matrix):
    row, col = matrix.shape
    row2, col2 = row // 2, col // 2
    return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:]


def strassen(x, y):
    if len(x) == 1:
        return x * y
    a, b, c, d = split(x)
    e, f, g, h = split(y)
    p1 = strassen(a, f - h)
    p2 = strassen(a + b, h)
    p3 = strassen(c + d, e)
    p4 = strassen(d, g - e)
    p5 = strassen(a + d, e + h)
    p6 = strassen(b - d, g + h)
    p7 = strassen(a - c, e + f)
    c11 = p5 + p4 - p2 + p6
    c12 = p1 + p2
    c21 = p3 + p4
    c22 = p1 + p5 - p3 - p7
    c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))

    return c

def strassen_mult(mat1, mat2, n):
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    return strassen(mat1, mat2)

def verify_result(mat1, mat2, computed_result):
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    computed_result = np.array(computed_result)
    reference_result = np.matmul(mat1, mat2)
    return np.array_equal(computed_result, reference_result)

def run_algs(algs, sizes, trials, mn = 1, mx = 10):
    dict_algs = {}
    for alg in algs:
        dict_algs[alg.__name__] = {}
    for size in sizes:
        for alg in algs:
            dict_algs[alg.__name__][size] = 0
        for trial in range(1, trials + 1):
            mat1 = random_matrix(mn, mx, size, size)
            mat2 = random_matrix(mn, mx, size, size)
            for alg in algs:
                start_time = time.time()
                prod = alg(mat1, mat2, size)
                if size == sizes[0]:
                    print(alg.__name__)
                    print(np.array(prod))
                    print("Verification:", "PASS" if verify_result(mat1, mat2, prod) else "FAIL")
                end_time = time.time()
                net_time = end_time - start_time
                dict_algs[alg.__name__][size] += 1000 * net_time
    return dict_algs

def main():
    assn = "Assignment03"
    sizes = [4, 16, 64]
    algs = [numpy_mult, listcomp_mult, simple_mult, divide_conq_mult, strassen_mult]
    trials = 1
    dict_algs = run_algs(algs, sizes, trials)
    as2.print_times(dict_algs, assn + ".txt")
    as2.plot_times(dict_algs, sizes, trials, algs, assn + ".png")

if __name__ == "__main__":
    main()
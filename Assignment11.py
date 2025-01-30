# Analysis of Algorithms (CSCI 323)
# Winter 2025
# Assignment 11 - Estimate, Evaluate and Rank Recurrence

import inspect
import math
import texttable
import matplotlib.pyplot as plt

# [1] Function to obtain the content of a function

def func_body(f):
    body = inspect.getsource(f)  # gets the code
    idx = body.index("return")  # get the part after the word return
    return body[7 + idx:].strip()

# [2] Create an empty dictionary to store intermediate results and a helper function ff to efficiently run a function f for input n:

dict_funcs = {}

def ff(f, n):
    func_name = f.__name__
    if func_name not in dict_funcs:
        dict_funcs[func_name] = {}
    dict_func = dict_funcs[func_name]
    if n not in dict_func:
        dict_func[n] = f(f, n)
    return dict_func[n]

# [3] Define recurrence functions

def f1_merge_sort(f, n):
    return 0 if n == 1 else 2 * ff(f, int(n/2)) + n

def f2_factorial(f, n):
    return 1 if n == 0 else n * ff(f, n-1)

def f3_fibonacci(f, n):
    return 0 if n == 0 else (1 if n == 1 else ff(f, n-1) + ff(f, n-2))

def f4_linear(f, n):
    return 0 if n == 0 else ff(f, n-1) + 1

def f5_quadratic(f, n):
    return 0 if n == 0 else ff(f, n-1) + 2*n

def f6_cubic(f, n):
    return 0 if n == 0 else ff(f, n-1) + n**2

def f7_exponential(f, n):
    return 1 if n == 0 else 2 * ff(f, n-1)

def f8_logarithmic(f, n):
    return 0 if n == 1 else ff(f, n//2) + 1

def f9_polylogarithmic(f, n):
    return 0 if n == 1 else ff(f, n//2) + int(math.log(n, 2))

def f10_sqrt(f, n):
    return 0 if n == 0 else ff(f, n-1) + int(math.sqrt(n))

def f11_factorial_growth(f, n):
    return 1 if n == 0 else n * ff(f, n-1)

def print_table(title, headers, alignments, data):
    table_obj = texttable.Texttable()
    table_obj.set_cols_align(alignments)
    table_obj.add_rows([headers] + data)
    print(title)
    print(table_obj.draw())
    print()
    return table_obj.draw()

# [4] Helper function to call and print results

def call_and_print(data, func, n):
    result = ff(func, n)
    data.append([func.__name__, func_body(func), n, result, math.log(result, 10) if result > 0 else 0])

# [5] Write terminal output to a file

def write_output_to_file(filename, table_text):
    with open(filename, "w") as f:
        f.write(table_text)

# [6] Plot bar graph

def plot_bar_graph(data):
    # Use logarithmic values for F(n) to handle large numbers
    labels = [f"{row[0]} (n={row[2]})" for row in data]
    values = [math.log(row[3], 10) if row[3] > 0 else 0 for row in data]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.xlabel("Recurrence Functions")
    plt.ylabel("log10(F(n))")  # Logarithmic scale
    plt.title("Evaluation of Recurrence Functions (Log Scale)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("Assignment11.png")
    plt.show()


# [7] Main function

def main():
    assn = "Assignment11"
    data = []
    funcs = [
        f1_merge_sort, f2_factorial, f3_fibonacci, f4_linear, f5_quadratic,
        f6_cubic, f7_exponential, f8_logarithmic, f9_polylogarithmic, f10_sqrt,
        f11_factorial_growth
    ]
    for func in funcs:
        for n in [10, 50, 100]:
            call_and_print(data, func, n)

    headers = ["Name", "Function", "n", "F(n)", "log F(n)"]
    alignments = ["l", "l", "r", "r", "r"]
    title = "Evaluation of Functions"

    table_text = print_table(title, headers, alignments, data)

    # Write terminal output to a text file
    write_output_to_file(f"{assn}.txt", table_text)

    # Plot bar graph
    plot_bar_graph(data)

if __name__ == "__main__":
    main()

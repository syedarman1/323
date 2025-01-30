# Analysis of Algorithms (CSCI 323)
# Winter 2025
# Assignment 7 - Graphs and Graph Algorithms

import time
from random import random

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


# [1] Define a function read_graph(file_name) that reads a graph from a text file and returns an adjacency/cost matrix.
def read_graph(file_name):
    with open(file_name) as file:
        lines = file.readlines()
        graph = [[int(x) for x in line.strip().split(" ")] for line in lines]
    return graph


# [2] Define a function adjacency_table(matrix) that accepts a graph as an adjacency/cost matrix and returns an adjacency/cost table.
def adjacency_table(matrix):
    n = len(matrix)
    table = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != 0:
                table[i].append(j)
    return table


# [3] Define a function edge_set(matrix) that accepts a graph as an adjacency/cost matrix and returns an edge/cost set.
def edge_set(matrix, directed):
    n = len(matrix)
    edges = set()
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != 0:
                if directed or i < j:
                    edges.add((i, j))
    return edges


# [4] Define a function edge_map(matrix) that accepts a graph as an adjacency/cost matrix and returns an edge/cost set.
def edge_map(matrix):
    n = len(matrix)
    map = {}
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != 0:
                map[(i, j)] = 1
    return map


def is_undirected(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True


# [6] Define a function random_graph(size, max_cost, p=1) that generates a graph with size edges, where each edge (except loops/self-edges) is assigned a random integer cost between 1 and max_cost.
# The additional parameter p represents the probability that there should be an edge between a given pair of vertices.
def random_graph(n, directed, p):
    matrix = [[1 if random() < p else 0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 0
    if not directed:
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j] = matrix[j][i]
    return matrix


# [7] Define functions that traverse the graph in these two standard orderings
# Breadth-First Search (BFS)
def bfs(matrix, s, visited):
    q = []
    n = len(matrix)
    visited.append(s)
    q.append(s)
    while q:
        curr = q.pop(0)
        for j in range(n):
            if matrix[curr][j] != 0 and j not in visited:
                visited.append(j)
                q.append(j)


def bfs_main(matrix):
    visited = []
    for i in range(len(matrix)):
        if i not in visited:
            bfs(matrix, i, visited)
    return visited


# Depth-First Search (DFS)
def dfs(matrix, s, visited):
    n = len(matrix)
    visited.append(s)
    for j in range(n):
        if matrix[s][j] != 0 and j not in visited:
            dfs(matrix, j, visited)


def dfs_main(matrix):
    visited = []
    for i in range(len(matrix)):
        if i not in visited:
            dfs(matrix, i, visited)
    return visited


def print_console_and_file(filename, *text):
    print(*text)
    print(*text, file=open(filename, "a"))


def print_adjacency_matrix(matrix, filename):
    n = len(matrix)
    labels = [str(i) for i in range(n)]

    print_console_and_file(filename, "Graph Adjacency Matrix:")
    print_console_and_file(filename, "    " + "  ".join(f"{label:4}" for label in labels))
    for i in range(n):
        print_console_and_file(filename, labels[i] + "  ".join(f"{matrix[i][j]:4}" for j in range(n)))
    print_console_and_file(filename)


def print_adjacency_table(table, filename):
    n = len(table)

    print_console_and_file(filename, "Graph Adjacency Table:")
    for i in range(n):
        print_console_and_file(filename, i, "-->", table[i])
    print_console_and_file(filename)


# [10] Define a function draw_graph(graph) that draws a graph
def draw_graph(edges, directed, filename):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, k=0.7)

    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='white')
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r', arrows=directed, arrowsize=10)

    plt.savefig(filename)
    plt.show()


def measure_time(alg, trials, *args):
    start_time = time.time()
    for _ in range(trials):
        result = alg(*args)
    return result, time.time() - start_time


def plot_times(dict_algs, sizes, trials, algs, file_name):
    alg_num = 0
    plt.xticks([j for j in range(len(sizes))], [str(size) for size in sizes])
    for alg in algs:
        alg_num += 1
        d = dict_algs[alg.__name__]
        x_axis = [j + 0.05 * alg_num for j in range(len(sizes))]
        y_axis = [d[i] for i in sizes]
        plt.bar(x_axis, y_axis, width=0.05, alpha=0.75, label=alg.__name__)
    plt.legend()
    plt.title("Runtime of algorithms")
    plt.xlabel("Number of elements")
    plt.ylabel("Time for " + str(trials) + " trials (ms)")
    plt.savefig(file_name)
    plt.show()


def print_times(dict_algs, filename):
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)
    df = pd.DataFrame.from_dict(dict_algs).T
    print_console_and_file(filename, df)


def do_graph(matrix, desc, image_filename, output_filename):

    table = adjacency_table(matrix)
    directed = not is_undirected(matrix)
    edges = edge_set(matrix, directed)
    map_edges = edge_map(matrix)

    print_console_and_file(output_filename, desc)
    print_console_and_file(output_filename)

    print_adjacency_matrix(matrix, output_filename)
    print_adjacency_table(table, output_filename)

    print_console_and_file(output_filename, "Edge Set:")
    print_console_and_file(output_filename, edges)
    print_console_and_file(output_filename)

    print_console_and_file(output_filename, "Edge Map:")
    print_console_and_file(output_filename, map_edges)
    print_console_and_file(output_filename)

    trial_count = 100000
    bfs, bfs_time = measure_time(bfs_main, trial_count, matrix)
    print_console_and_file(output_filename, "BFS Traversal:", bfs)

    dfs, dfs_time = measure_time(dfs_main, trial_count, matrix)
    print_console_and_file(output_filename, "DFS Traversal:", dfs)
    print_console_and_file(output_filename, "\n")

    draw_graph(edges, directed, image_filename)

    dict_algs = {bfs_main.__name__: {len(bfs): bfs_time}, dfs_main.__name__: {len(dfs): dfs_time}}
    print_times(dict_algs, output_filename)
    plot_times(dict_algs, [len(matrix)], trial_count, [bfs_main, dfs_main], f"{output_filename.split('.')[0]}_times.png")


def main():
    assn = "assignment07"
    output = f"{assn}.txt"
    open(output, "w").close()

    matrix1 = read_graph(f"{assn}_graph.txt")
    do_graph(matrix1, "Graph from file:", f"{assn}_graph_from_file.png", output)

    matrix2 = random_graph(5, False, 0.8)
    do_graph(matrix2, "Random Graph:", f"{assn}_random_graph.png", output)


if __name__ == "__main__":
    main()

# Analysis of Algorithms (CSCI 323)
# Winter 2025
# Assignment 9 - Shortest Path Algorithms

from random import randint, random
from time import time

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

import Assignment02 as as2


INF = 9999


# [1] Define a function floyd_apsp(graph) that solves APSP for a graph using Floyd's dynamic programming algorithm.
def floyd_apsp(matrix, dist, pred, n):
    n = len(matrix)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    pred[i][j] = pred[k][j]


# [2] Define a function bellman_ford_sssp(es, n, src) that takes an edge-set of the graph, the size of the graph, and a starting point, and solves SSSP using the Bellman Ford dynamic programming algorithm.
def bellman_ford_sssp(edges, src, dist, pred, n):
    for _ in range(n):
        for edge in edges:
            u, v, wt = edge
            if dist[src][u] + wt < dist[src][v]:
                dist[src][v] = dist[src][u] + wt
                pred[src][v] = u
    return dist


def bellman_ford_apsp(matrix, dist, pred, n):
    edges = edge_set(matrix)
    for i in range(n):
        bellman_ford_sssp(edges, i, dist, pred, n)


# [4] Define a function dijkstra_sssp_matrix(graph, src) that takes a graph and a starting point, and solves SSSP using Dijsktra's greedy SSSP algorithm, assuming an adjacency matrix and minimization over an array.
def min_dist(dist, visited, src, n):
    min = INF
    min_index = -1
    for v in range(n):
        if dist[src][v] < min and not visited[v]:
            min = dist[src][v]
            min_index = v
    return min_index


def dijkstra_sssp(matrix, src, dist, pred, n):
    visited = [False] * n
    for i in range(n):
        x = min_dist(dist, visited, src, n)
        visited[x] = True
        for y in range(n):
            if not visited[y] and dist[src][y] > dist[src][x] + matrix[x][y]:
                dist[src][y] = dist[src][x] + matrix[x][y]
                pred[src][y] = x


def dijkstra_apsp(matrix, dist, pred, n):
    for i in range(n):
        dijkstra_sssp(matrix, i, dist, pred, n)


def random_weighted_graph(size, min_w, max_w, p_edge):
    return [[randint(min_w, max_w) if i != j and random() < p_edge else INF for j in range(size)] for i in range(size)]


def init_dist_pred(matrix, n):
    pred = [[i for _ in range(n)] for i in range(n)]
    dist = [[matrix[i][j] for j in range(n)] for i in range(n)]
    for i in range(n):
        dist[i][i] = 0
        pred[i][i] = -1
    return dist, pred


def edge_set(matrix):
    n = len(matrix)
    edges = []
    for i in range(n):
        for j in range(n):
            if matrix[i][j] < INF:
                edges.append((i, j, matrix[i][j]))
    return edges


def draw_graph(edges, directed, filename):
    G = nx.DiGraph()
    # Convert edges to the correct format for weighted edges
    processed_edges = [(u, v, {'weight': w}) for u, v, w in edges]
    G.add_edges_from(processed_edges)
    pos = nx.spring_layout(G, k=0.8, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='white')
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r', arrows=directed, arrowsize=10)

    plt.savefig(filename)
    plt.show()


def print_console_and_file(filename, *text):
    print(*text)
    print(*text, file=open(filename, "a"))


def print_adjacency_matrix(matrix, filename, desc):
    n = len(matrix)
    labels = [str(i) for i in range(n)]

    print_console_and_file(filename, desc)
    print_console_and_file(filename, "        " + "  ".join(f"{label:4}" for label in labels))
    for i in range(n):
        print_console_and_file(filename, f"{labels[i]}    " + "  ".join(f"{matrix[i][j]:4}" for j in range(n)))
    print_console_and_file(filename)


def print_times(dict_algs, filename):
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)
    df = pd.DataFrame.from_dict(dict_algs).T
    print_console_and_file(filename, "Time (ms) for each algorithm:")
    print_console_and_file(filename, df)


def run_algs(algs, sizes, trials, assn):
    dict_algs = {}
    for alg in algs:
        dict_algs[alg.__name__] = {}
    for size in sizes:
        for alg in algs:
            dict_algs[alg.__name__][size] = 0
        for trial in range(1, trials + 1):
            matrix = random_weighted_graph(size, 10, 99, 0.3)
            for alg in algs:
                start_time = time()
                dist, pred = init_dist_pred(matrix, size)
                alg(matrix, dist, pred, size)
                end_time = time()
                net_time = end_time - start_time
                dict_algs[alg.__name__][size] += 1000 * net_time
                if size == sizes[0]:
                    print_console_and_file(f"{assn}.txt", "\n", alg.__name__)
                    print_adjacency_matrix(dist, f"{assn}.txt", "Shortest Path Distance Matrix:")
                    print_adjacency_matrix(pred, f"{assn}.txt", "Predecessor Matrix:")
                    draw_graph(edge_set(matrix), True, f"{assn}_graph.png")

    return dict_algs


def main():
    assn = "assignment09"
    open(f"{assn}.txt", "w").close()

    algs = [floyd_apsp, dijkstra_apsp, bellman_ford_apsp]
    trials = 1
    sizes = [10, 20, 30, 40]
    dict_algs = run_algs(algs, sizes, trials, assn)

    print_times(dict_algs, f"{assn}.txt")
    as2.plot_times(dict_algs, sizes, trials, algs, f"{assn}.png")


if __name__ == '__main__':
    main()

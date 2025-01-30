# Analysis of Algorithms (CSCI 323)
# Winter 2025
# Assignment 8 - Minimum Spanning Tree Algorithms

from random import randint, random
from time import time

import pandas as pd

import Assignment02 as as2


INF = 9999


def random_weighted_graph(size, min_w, max_w, p_edge, directed):
    matrix = [[randint(min_w, max_w) if i != j and random() < p_edge else INF for j in range(size)] for i in range(size)]
    if not directed:
        matrix = [[matrix[i][j] if i < j else matrix[j][i] for j in range(size)] for i in range(size)]

    return matrix


def min_dist(dist, visited, src, n):
    min = INF
    min_index = -1
    for v in range(n):
        if dist[src][v] < min and not visited[v]:
            min = dist[src][v]
            min_index = v
    return min_index


def min_key(key, mstSet, n):
    min_k = INF
    min_idx = -1

    for v in range(n):
        if key[v] < min_k and not mstSet[v]:
            min_k = key[v]
            min_idx = v

    return min_idx


def prim_mst(matrix, n):
    key = [INF] * n
    parent = [None] * n
    mstSet = [False] * n

    key[0] = 0
    parent[0] = -1

    for _ in range(n):
        u = min_key(key, mstSet, n)
        mstSet[u] = True

        for v in range(n):
            if not mstSet[v] and key[v] > matrix[u][v]:
                key[v] = matrix[u][v]
                parent[v] = u

    mst = [(parent[i], i, matrix[i][parent[i]]) for i in range(1, n)]
    return mst


def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])


def union(parent, rank, x, y):
    if rank[x] < rank[y]:
        parent[x] = y
    elif rank[x] > rank[y]:
        parent[y] = x
    else:
        parent[y] = x
        rank[x] += 1


def kruskal_mst(matrix, n):
    result = []
    i = 0
    e = 0
    edges = edge_set(matrix, False)
    edges = sorted(edges, key=lambda item: item[2])

    parent = []
    rank = []
    for node in range(n):
        parent.append(node)
        rank.append(0)

    while e < n - 1:
        u, v, w = edges[i]
        i += 1
        x = find(parent, u)
        y = find(parent, v)

        if x != y:
            e += 1
            result.append((u, v, w))
            union(parent, rank, x, y)

    return result


def mst_cost(mst):
    return sum([mst[i][2] for i in range(len(mst))])


def edge_set(matrix, directed):
    n = len(matrix)
    edges = []
    for i in range(n):
        for j in range(n):
            if matrix[i][j] < INF:
                if directed or i < j:
                    edges.append((i, j, matrix[i][j]))
    return edges


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
    print_console_and_file(filename, "\nTime (ms) for each algorithm:\n", df)


def run_algs(algs, sizes, trials, assn):
    dict_algs = {}
    for alg in algs:
        dict_algs[alg.__name__] = {}
    for size in sizes:
        for alg in algs:
            dict_algs[alg.__name__][size] = 0
        for trial in range(1, trials + 1):
            matrix = random_weighted_graph(size, 10, 99, 1, False)
            for alg in algs:
                start_time = time()
                mst = alg(matrix, size)
                end_time = time()
                net_time = end_time - start_time
                dict_algs[alg.__name__][size] += 1000 * net_time
                if size == sizes[0]:
                    print_console_and_file(f"{assn}.txt", alg.__name__, mst_cost(mst), mst)

    return dict_algs


def main():
    assn = "assignment08"
    open(f"{assn}.txt", "w").close()

    algs = [prim_mst, kruskal_mst]
    trials = 10
    sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    dict_algs = run_algs(algs, sizes, trials, assn)

    print_times(dict_algs, f"{assn}.txt")
    as2.plot_times(dict_algs, sizes, trials, algs, f"{assn}_times.png")


if __name__ == '__main__':
    main()

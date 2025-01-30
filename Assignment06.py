# Analysis of Algorithms (CSCI 323)
# Winter 2025
# Assignment 6 - Optimal Triangulation Problem

from time import time
import matplotlib.pyplot as plt
import Assignment04 as as4
import Assignment02 as as2
import math



#[1] Generate a set of n points. Use an n somewhat larger than what you actually want for the triangulation, since points will get eliminated when forming the convex hull / polygon.


# [2] Find the convex hull of the points. This will give you a reduced set of points.


#[3] Compute the optimal triangulation using recursion.
MAX = 100000


# Distance formula between two points p1 and p2
def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def cost(points, i, j, k):
    return dist(points[i], points[j]) + dist(points[j], points[k])

def ot_rec_main(points):
    n = len(points)
    tri = [[-1] * n for i in range(n)]
    ot = ot_rec(points, 0, n - 1, tri)
    return ot, tri


def ot_rec(points, i, j, tri):
    if j < i + 2:
        return 0
    res = MAX
    for k in range(i + 1, j):
        temp = ot_rec(points, i, k, tri) + ot_rec(points, k, j, tri) + cost(points, i, k, j)
        if temp < res:
            res = temp
            tri[i][j] = k
    return res

# [4] Compute the optimal triangulation using memoization.
def ot_mem(points, i, j, mem, tri):
    if mem[i][j] < 0:
        if j < i + 2:
            mem[i][j] = 0
        else:
            res = MAX
            for k in range(i + 1, j):
                temp = ot_mem(points, i, k, mem, tri) + ot_mem(points, k, j, mem, tri) + cost(points, i, k, j)
                if temp < res:
                    res = temp
                    tri[i][j] = k
            mem[i][j] = res
    return mem[i][j]


def ot_mem_main(points):
    n = len(points)
    mem = [[-1] * n for i in range(n)]
    tri = [[-1] * n for i in range(n)]
    ot = ot_mem(points, 0, n - 1, mem, tri)
    return ot, tri

# [5] Compute the optimal triangulation using dynamic programming.
def ot_dp_main(points):
    n = len(points)
    tri = [[-1] * n for i in range(n)]
    ot = ot_dp(points, n, tri)
    return ot, tri


def ot_dp(points, n, tri):
    n = len(points)
    if n < 3:
        return 0
    table = [[0.0] * n for _ in range(n)]
    gap = 0
    while gap < n:
        i = 0
        j = gap
        while j < n:
            if j < i + 2:
                table[i][j] = 0.0
            else:
                table[i][j] = MAX
                k = i + 1
                while k < j:
                    val = table[i][k] + table[k][j] + cost(points, i, j, k)
                    if table[i][j] > val:
                        table[i][j] = val
                    k += 1
            i += 1
            j += 1
        gap += 1
    return table[0][n - 1]

#[6] Verify algorithmically that the answers using all three approaches (Tasks 3, 4, 5) are the same.

#[7] Fit the three algorithms into our infrastructure from earlier assignments so all algorithms get executed for random data and their times get tracked, tabulated and plotted.
def run_algs(algs, sizes, trials):
    dict_algs = {}
    for alg in algs:
        dict_algs[alg.__name__] = {}
    for size in sizes:
        for alg in algs:
            dict_algs[alg.__name__][size] = 0
        for trial in range(1, trials + 1):
            points = as4.generate_points(size, -10, 10)
            #sorted_points = as4.sort_points(points) convex hull function will sort the points
            cv = as4.convex_hull(points)
            for alg in algs:
                start_time = time()
                ot, tri = alg(cv)
                end_time = time()
                net_time = end_time - start_time
                dict_algs[alg.__name__][size] += 1000 * net_time
                if size == sizes[0]:
                    print("\n", alg.__name__, round(ot, 4))
                    print(tri)
                    as4.draw_points(cv, "g", f"Assignment6-cv-{size}.png")
                    draw_triangulated_convex_hull(cv, tri, f"Assignment6-tri-{size}.png")
    return dict_algs


def draw_triangulated_convex_hull(hull, tri, file_name):
    n = len(hull)
    x = [hull[i][0] for i in range(n)]
    y = [hull[i][1] for i in range(n)]
    plt.plot(x, y, 'ro')
    plt.axis('equal')
    for i in range(n - 1):
        plt.plot(x[i:i + 2], y[i:i + 2], 'bo-')
    xx = [x[n - 1], x[0]]
    yy = [y[n - 1], y[0]]
    plt.plot(xx, yy, 'bo-')
    for i in range(n):
        for j in range(n):
            if tri[i][j] >= 0:
                a = i
                b = tri[i][j]
                xx = [hull[a][0], hull[b][0]]
                yy = [hull[a][1], hull[b][1]]
                plt.plot(xx, yy, 'go-')
    plt.savefig(file_name)
    plt.show()


def main():
    assn = "Assignment06"
    algs = [ot_rec_main, ot_dp_main, ot_mem_main]
    sizes = [20]
    trials = 1
    dict_algs = run_algs(algs, sizes, trials)
    as2.print_times(dict_algs, f"{assn}.txt")
    as2.plot_times(dict_algs, sizes, trials, algs, f"{assn}.png")


if __name__ == "__main__":
    main()

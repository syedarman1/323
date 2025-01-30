# Analysis of Algorithms (CSCI 323)
# Winter 2025
# Assignment 10 - String Search algorithms

from random import choices, randint
from time import time

import texttable

import Assignment02 as as2


# [1] Define functions that implement these sub-string search algorithms:
# Native Search that wraps the built-in string-search capability of your programming language
# Brute Force - see https://www.geeksforgeeks.org/naive-algorithm-for-pattern-searching/
# Rabin-Karp - see https://www.geeksforgeeks.org/rabin-karp-algorithm-for-pattern-searching
# Rabin-Karp Randomized - (use multiple hash functions with random moduli)
# Knuth-Morris-Pratt - see https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching
# Boyer-Moore - see https://www.geeksforgeeks.org/boyer-moore-algorithm-for-pattern-searching
def random_string(size, alphabet):
    return ''.join(choices(alphabet, k=size))


def native_search(text, text_len, pattern, pattern_len):
    try:
        return text.index(pattern)
    except ValueError:
        return -1


def brute_force(text, text_len, pattern, pattern_len):
    for curr_index in range(text_len - pattern_len + 1):
        match_index = 0
        while match_index < pattern_len and text[curr_index + match_index] == pattern[match_index]:
            match_index += 1
        if match_index == pattern_len:
            return curr_index
    return -1


def rabin_karp(text, text_len, pattern, pattern_len):
    prime = 101
    num_char = 256

    pattern_hash = 0
    text_hash = 0
    hash_weight = 1

    for _ in range(pattern_len - 1):
        hash_weight = (hash_weight * num_char) % prime

    for i in range(pattern_len):
        pattern_hash = (num_char * pattern_hash + ord(pattern[i])) % prime
        text_hash = (num_char * text_hash + ord(text[i])) % prime

    for curr_index in range(text_len - pattern_len + 1):
        if pattern_hash == text_hash:
            matched_len = 0
            for j in range(pattern_len):
                if text[curr_index + j] != pattern[j]:
                    break
                matched_len += 1

            if matched_len == pattern_len:
                return curr_index

        if curr_index < text_len - pattern_len:
            text_hash = (num_char * (text_hash - ord(text[curr_index]) * hash_weight) + ord(text[curr_index + pattern_len])) % prime

    return -1


def construct_lps(pattern, pattern_len, lps):
    longest_len = 0

    curr_index = 1
    while curr_index < pattern_len:
        if pattern[curr_index] == pattern[longest_len]:
            longest_len += 1
            lps[curr_index] = longest_len
            curr_index += 1

        else:
            if longest_len != 0:
                longest_len = lps[longest_len - 1]
            else:
                lps[curr_index] = 0
                curr_index += 1


def knuth_morris_pratt(text, text_len, pattern, pattern_len):

    lps = [0] * pattern_len
    construct_lps(pattern, pattern_len, lps)

    curr_txt_index = 0
    curr_pat_ind = 0

    while curr_txt_index < text_len:
        if text[curr_txt_index] == pattern[curr_pat_ind]:
            curr_txt_index += 1
            curr_pat_ind += 1

            if curr_pat_ind == pattern_len:
                return curr_txt_index - curr_pat_ind
        else:
            if curr_pat_ind != 0:
                curr_pat_ind = lps[curr_pat_ind - 1]
            else:
                curr_txt_index += 1

    return -1


def bad_char_heuristic(string, size):
    NO_OF_CHARS = 256

    badChar = [-1] * NO_OF_CHARS

    for i in range(size):
        badChar[ord(string[i])] = i

    return badChar


def boyer_moore(text, text_len, pattern, pattern_len):

    badChar = bad_char_heuristic(pattern, pattern_len)

    curr_txt_ind = 0
    while curr_txt_ind <= text_len - pattern_len:
        curr_pat_ind = pattern_len - 1
        while curr_pat_ind >= 0 and pattern[curr_pat_ind] == text[curr_txt_ind + curr_pat_ind]:
            curr_pat_ind -= 1

        if curr_pat_ind < 0:
            return curr_txt_ind
        else:
            curr_txt_ind += max(1, curr_pat_ind - badChar[ord(text[curr_txt_ind + curr_pat_ind])])

    return -1


def run_algs(algs, sizes, trials):
    dict_algs = {}
    data = []
    for alg in algs:
        dict_algs[alg.__name__] = {}
    for size in sizes:
        for alg in algs:
            dict_algs[alg.__name__][size] = 0
        for trial in range(1, trials + 1):
            pattern_len = 10
            text = random_string(size, "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            idx = randint(0, size - pattern_len)
            pattern = text[idx:idx + pattern_len]
            for alg in algs:
                start_time = time()
                idx_found = alg(text, size, pattern, pattern_len)
                end_time = time()
                net_time = end_time - start_time
                dict_algs[alg.__name__][size] += 1000 * net_time
                data.append([alg.__name__, size, pattern_len, pattern, idx_found, net_time])

    return dict_algs, data


def print_table(title, headers, alignments, data, filename):
    tableObj = texttable.Texttable()
    tableObj.set_cols_align(alignments)
    tableObj.add_rows([headers] + data)

    print(f"\n\n{title}\n{tableObj.draw()}")
    print(f"\n\n{title}\n{tableObj.draw()}", file=open(filename, "a"))


def main():
    headers = ["Algorithm", "Len text", "Len pattern", "pattern", "idx_found", "time"]
    alignments = ["l", "r", "r", "l", "r", "r"]
    title = "String Search Results"

    assn = "assignment10"
    trials = 10
    algs = [native_search, brute_force, rabin_karp, knuth_morris_pratt, boyer_moore]
    sizes = [10, 100, 1000, 10000, 100000, 1000000]

    dict_algs, data = run_algs(algs, sizes, trials)
    as2.print_times(dict_algs, f"{assn}.txt")
    as2.plot_times(dict_algs, sizes, trials, algs, f"{assn}.png")
    print_table(title, headers, alignments, data, f"{assn}.txt")


if __name__ == "__main__":
    main()

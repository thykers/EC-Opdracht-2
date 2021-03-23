import numpy as np
# import list
import copy
import cProfile
import random
import time


class Node:
    index = 0
    gain = 0
    prev = None
    next = None
    links = []
    is_in_bucket = False

    def __init__(self, index, links):
        self.index = index
        self.links = links


class NodeDoubleLinkedList(object):

    def __init__(self, gain):
        self.head = None
        self.tail = None
        self.count = 0
        self.gain = gain

    def append(self, node):
        # Append an item 
        if self.head is None:
            self.head = node
            self.tail = self.head
            node.prev = None
        else:
            node.prev = self.tail
            self.tail.next = node
            self.tail = node
        node.next = None
        node.is_in_bucket = True
        self.count += 1

    def pop(self):
        if self.tail is not None:
            tmp = self.tail
            if self.tail.prev is not None:  # There is at least one more item before the tail
                self.tail.prev.next = None
                self.tail = self.tail.prev
            else:
                self.tail = None
                self.head = None
            self.count -= 1
            tmp.is_in_bucket = False
            return tmp
        else:
            print("popping empty list")

    def exists(self, node):
        if self.head is None:
            return False
        current = self.head
        while True and current is not None:
            if current.index == node.index:
                return True
            if current.next is None:
                print("Node does not exist")
                return False
            current = current.next
        return True

    def remove_node(self, node):
        if node.is_in_bucket:
            node.is_in_bucket = False
            self.count -= 1
            if node.prev is None and node.next is None:
                self.tail = None
                self.head = None
                return
            if node.next is None:  # Node is last item
                self.tail = node.prev
            else:
                node.next.prev = node.prev
            if node.prev is None:  # Node is first item
                self.head = node.next
            else:
                node.prev.next = node.next
        else:
            print("Removing node that is not in a bucket")


class Graph:
    nodes_list = []
    maxLinks = 0

    def __init__(self):
        self.read_graph()

    def read_graph(self):
        f = open("Graph500.txt", "r")
        for index, line in enumerate(f):
            split = line.split()
            links = list(map(int, split[3:]))
            self.maxLinks = max(len(links), self.maxLinks)
            self.nodes_list.append(Node(index + 1, links))


class Model:
    local_search_count_calls = 0
    max_local_search_calls = 1000

    def __init__(self):
        self.Graph = Graph()
        self.maxLinks = self.Graph.maxLinks

    def calculate_cuts(self, solution):
        cut = 0
        for node in self.Graph.nodes_list:
            own_value = solution[node.index - 1]
            for link in node.links:
                if own_value != solution[link - 1]:
                    cut += 1
        return int(cut / 2)

    def generate_random_solution(self):
        node_amount = len(self.Graph.nodes_list)
        solution = np.repeat(0, node_amount)
        solution[int(node_amount / 2):] = np.repeat(1, int(node_amount / 2))
        # np.random.seed(1)
        np.random.shuffle(solution)
        return solution

    def get_gain(self, i):
        return \
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7,
             -6, -5, -4, -3, -2, -1][i]

    def generate_buckets(self):
        left = [NodeDoubleLinkedList(self.get_gain(i)) for i in range(self.maxLinks * 2 + 1)]
        right = [NodeDoubleLinkedList(self.get_gain(i)) for i in range(self.maxLinks * 2 + 1)]
        return [left, right]

    def calculate_gain_node(self, solution, node):
        gain = 0
        own_value = solution[node.index - 1]
        for link in node.links:
            if own_value == solution[link - 1]:
                gain -= 1
            else:
                gain += 1
        node.gain = gain

    def calculate_gains_graph(self, solution, gain_buckets):
        for node in self.Graph.nodes_list:
            own_value = solution[node.index - 1]
            self.calculate_gain_node(solution, node)
            if own_value == 0:
                gain_buckets[0][node.gain].append(node)
            else:
                gain_buckets[1][node.gain].append(node)
        return gain_buckets

    def swap_best(self, solution, score, bucket_index, gain_buckets):
        for i in range(self.maxLinks, -self.maxLinks - 1, -1):
            if gain_buckets[bucket_index][i].count > 0:
                selected_node = gain_buckets[bucket_index][i].pop()
                solution[selected_node.index - 1] = 1 - solution[selected_node.index - 1]

                # Update neighbours
                for neighbour in [self.Graph.nodes_list[link - 1] for link in selected_node.links]:
                    # Remove from the bucket it is in
                    if neighbour.is_in_bucket:
                        if solution[neighbour.index - 1] == 0:
                            gain_buckets[0][neighbour.gain].remove_node(neighbour)
                        else:
                            gain_buckets[1][neighbour.gain].remove_node(neighbour)

                        # Recalculate gain and add to the correct bucket
                        self.calculate_gain_node(solution, neighbour)
                        if solution[neighbour.index - 1] == 0:
                            gain_buckets[0][neighbour.gain].append(neighbour)
                        else:
                            gain_buckets[1][neighbour.gain].append(neighbour)
                score -= selected_node.gain
                break
        return solution, score

    def print_partition(self, solution, score):
        l, r = [], []
        for i, v in enumerate(solution):
            if v == 0:
                l.append(i + 1)
            else:
                r.append(i + 1)
        print(f"left: {l}  |   right: {r}")
        print(f"Cuts: {score}")

    def fm_pass(self, solution):
        # initialize partition and buckets
        score = self.calculate_cuts(solution)

        empty_buckets = self.generate_buckets()
        gain_buckets = self.calculate_gains_graph(solution, empty_buckets)
        bucket_index = 0

        best_solution = copy.deepcopy(solution)  # Store a copy of the best solution
        best_score = score
        # print(f"Initial: {score}")

        # Run
        for i, node in enumerate(self.Graph.nodes_list):
            solution, score = self.swap_best(solution, score, bucket_index, gain_buckets)
            if score < best_score and i % 2 == 1:  # Only look for better score every two swaps,
                # print(f"new best score: {score}")  # because we need an equal amount of nodes in each group.
                best_solution = copy.deepcopy(solution)
                best_score = score
            bucket_index = 1 - bucket_index

        # print(f"Best score: {best_score}")
        self.local_search_count_calls += 1
        return best_solution, best_score

    def mls(self, convergence_criteria):
        solution = self.generate_random_solution()
        score = self.calculate_cuts(solution)

        mls_solution = copy.deepcopy(solution)
        mls_score = score

        while self.local_search_count_calls < self.max_local_search_calls:
            no_change_in_solution_count = 0
            run_solution = copy.deepcopy(solution)
            run_score = self.calculate_cuts(run_solution)
            while no_change_in_solution_count < convergence_criteria:
                new_solution, new_score = self.fm_pass(solution)

                if new_score >= run_score:
                    no_change_in_solution_count += 1
                elif new_score < score:
                    no_change_in_solution_count = 0
                    run_solution = copy.deepcopy(new_solution)
                    run_score = new_score

            if run_score < mls_score:
                mls_solution = copy.deepcopy(run_solution)
                mls_score = run_score

            solution = self.generate_random_solution() #mls

        # print(self.local_search_count_calls)
        print(mls_score)
        print(f"Cuts verification: {self.calculate_cuts(mls_solution)}")
        return mls_solution, mls_score

    def mutate_solution(self, solution, perturbation_size):
        mutated_solution = copy.deepcopy(solution)
        solution = copy.deepcopy(np.array([1 - bit if (random.uniform(0, 1)) <= perturbation_size else bit
                                           for bit in mutated_solution]))
        return solution

    def ils(self, perturbation_size, convergence_criteria):
        solution = self.generate_random_solution()
        score = self.calculate_cuts(solution)

        ils_solution = copy.deepcopy(solution)
        ils_score = score

        while self.local_search_count_calls < self.max_local_search_calls:
            no_change_in_solution_count = 0
            run_solution = copy.deepcopy(solution)
            run_score = self.calculate_cuts(run_solution)
            while no_change_in_solution_count < convergence_criteria:
                new_solution, new_score = self.fm_pass(solution)

                if new_score >= run_score:
                    no_change_in_solution_count += 1
                elif new_score < score:
                    no_change_in_solution_count = 0
                    run_solution = copy.deepcopy(new_solution)
                    run_score = new_score

            if run_score < ils_score:
                ils_solution = copy.deepcopy(run_solution)
                ils_score = run_score

            solution = self.mutate_solution(ils_solution, perturbation_size)

        # print(self.local_search_count_calls)
        print(ils_score)
        print(f"Cuts verification: {self.calculate_cuts(ils_solution)}")
        return ils_solution, ils_score

    def gls(self):
        ...


x = Model()

timer = time.time()
x.mls(5)
#x.ils(0.05, 5)
print(f"Elapsed: {time.time() - timer}")

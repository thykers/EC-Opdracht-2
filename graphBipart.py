import numpy as np
# import list
import copy
import cProfile
import random
import time
import networkx as nx
import matplotlib.pyplot as plt
import scipy
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
    max_local_search_calls = 10000

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
        if self.calculate_cuts(best_solution) != best_score:
            print("mistake")
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
        sampleSize = np.random.binomial(len(solution), perturbation_size * 2)

        indices_0 = set()
        while len(indices_0) <= sampleSize:
            index = random.randint(0, len(solution) - 1)
            if solution[index] == 0:
                indices_0.add(index)
        indices_1 = set()
        while len(indices_1) <= sampleSize:
            index = random.randint(0, len(solution) - 1)
            if solution[index] == 1:
                indices_1.add(index)
        mutated_solution = copy.deepcopy(solution)
        mutated_solution[list(indices_0)] = 1
        mutated_solution[list(indices_1)] = 0
        return copy.deepcopy(mutated_solution)

    def mutate_solution_2(self, solution, perturbation_size):
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
        if not self.valid_solution(ils_solution):
            print("error")
        return ils_solution, ils_score

    def hamming_distance(self, parent1, parent2):
        return sum(parent1 == parent2)
    
    def equal_swap(self, child, parent2):
        #Uniform crossover such that amount of bits is equal:
        crossoverProb = 0.5
        even = 0
        for i in range(len(child)):
            if child[i] != parent2[i]:
                if even == 0:
                    if (random.uniform(0, 1) <= crossoverProb):
                        if parent2[i] == 1:
                            even += 1
                        if parent2[i] == 0:
                            even -= 0
                        child[i] = parent2[i]
    
    def valid_solution(self, solution):
        return sum(solution) == int(len(solution) / 2)

    def gls(self, populationsize=50, convergence_criteria=5):
        population = []
        scores = []
        #Find population of 50 local optima
        for i in range(0, populationsize):
            no_change_in_solution_count = 0
            run_solution = self.generate_random_solution()
            run_score = self.calculate_cuts(run_solution)
            while no_change_in_solution_count < convergence_criteria:
                new_solution, new_score = self.fm_pass(run_solution)

                if new_score >= run_score:
                    no_change_in_solution_count += 1
                elif new_score < run_score:
                    no_change_in_solution_count = 0
                    run_solution = copy.deepcopy(new_solution)
                    run_score = new_score
            population.append(copy.deepcopy(run_solution))
            scores.append(run_score)
            run_solution = self.generate_random_solution()
        sortedIndices = list(range(populationsize))
        sortedIndices.sort(key=lambda x: scores[x])

        #Generate new child, do fm-pass until local optima and compete with worst solution.
        while self.local_search_count_calls < self.max_local_search_calls:
            parents = random.sample(population, 2)
            if self.hamming_distance(parents[0], parents[1]) > int(len(parents[0]) / 2):
                offspring1 = 1 - copy.deepcopy(parents[0])
            else:
                offspring1 = copy.deepcopy(parents[0])

            indices_1_to_0 = np.array([x for x in list(range(len(parents[0]))) if offspring1[x] != parents[1][x] and parents[1][x] == 0])
            indices_0_to_1 = np.array([x for x in list(range(len(parents[0]))) if offspring1[x] != parents[1][x] and parents[1][x] == 1])
            sampleSize = random.randint(0, min(len(indices_0_to_1), len(indices_1_to_0)))
            indices = np.concatenate((np.random.choice(indices_1_to_0, size=sampleSize, replace=False), np.random.choice(indices_0_to_1, size=sampleSize, replace=False)))
            offspring1[indices] = parents[1][indices]
            no_change_in_solution_count = 0

            run_solution = copy.deepcopy(offspring1)
            run_score = self.calculate_cuts(run_solution)
            while no_change_in_solution_count < convergence_criteria:
                new_solution, new_score = self.fm_pass(copy.deepcopy(run_solution))
                if self.calculate_cuts(new_solution) != new_score:
                    print("mistake")
                if new_score >= run_score:
                    no_change_in_solution_count += 1
                elif new_score < run_score:
                    no_change_in_solution_count = 0
                    run_solution = copy.deepcopy(new_solution)
                    run_score = new_score

            if run_score < scores[sortedIndices[-1]]:
                #replace worst solution
                if self.calculate_cuts(run_solution) != run_score:
                    print("mistake")
                population[sortedIndices[-1]] = copy.deepcopy(run_solution)
                scores[sortedIndices[-1]] = run_score
                sortedIndices.sort(key=lambda x: scores[x])
        print(scores[sortedIndices[0]])
        print(f"Cuts verification: {self.calculate_cuts(population[sortedIndices[0]])}")
        return population[sortedIndices[0]], scores[sortedIndices[0]]

    def get_color(self, node, solution):
        if solution[node.index - 1] == 1:
            return "red"
        else:
            return "blue"

    def draw_network(self, solution):
        nodes = [node.index for node in self.Graph.nodes_list]
        edges = set()
        for node in self.Graph.nodes_list:
            for link in node.links:
                edges.add((min(node.index, link), max(node.index, link)))
        print(edges)
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        position_dict = {}
        f = open("Graph500.txt", "r")
        for index, line in enumerate(f):
            positions = line.split()[1].replace('(', '').replace(')', '').split(',')
            x = float(positions[0])
            y = float(positions[1])
            position_dict[(index + 1)] = (x,y)
        node_colors = [self.get_color(node, solution) for node in self.Graph.nodes_list]
        nx.draw_networkx(g, pos=position_dict, node_color=node_colors)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()

    def ils_test(model):
        test_values = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5]
        best_score = 1000
        history = []
        for value in test_values[6:7]:
            print(f"Test Value: {value}")
            total_repeat_count = 0
            total_score = 0
            for i in range(1, 26):
                print(f"Run: {i}")
                solution, score, repeat_count = model.ils(value, 3)
                total_repeat_count += repeat_count
                total_score += score
            average_score = int(total_score / 25)
            average_repeat = int(repeat_count / 25)
            print(f"Average score: {average_score} -- Average repeat: {average_repeat}")
            history.append((average_score, average_repeat))
            if average_score <= best_score:
                best_score = average_score
            else:
                break
        print(history)

x = Model()
timer = time.time()
#x.hamming_distance(x.generate_random_solution(), x.generate_random_solution())
#x.mls(3)
#for i in range(10):
    #x.local_search_count_calls = 0
x.ils(0.05, 3)
#sol, score = x.gls(populationsize=50, convergence_criteria=5)
#x.draw_network(sol)
print(f"Elapsed: {time.time() - timer}")

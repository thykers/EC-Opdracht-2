import numpy as np
# import list
import copy
import cProfile


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
    def __init__(self):
        self.head = None
        self.tail = None
        self.count = 0

    def append(self, node):
        # Append an item 
        if self.head is None:
            self.head = node
            self.tail = self.head
        else:
            node.prev = self.tail
            self.tail.next = node
            self.tail = node
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

    def remove_node(self, node):
        if node.is_in_bucket:
            if node.prev is None and node.next is None:
                self.tail = None
                self.head = None
            if node.next is None:  # Node is last item
                self.tail = node.prev
            else:
                node.next.prev = node.prev

            if node.prev is None:  # Node is first item
                self.head = node.next
            else:
                node.prev.next = node.next
            node.is_in_bucket = False
            self.count -= 1
        else:
            print("Removing node that is not in a bucket")


class Graph:
    nodes = []
    maxLinks = 0

    def __init__(self):
        self.read_graph()

    def read_graph(self):
        f = open("Graph500.txt", "r")
        for index, line in enumerate(f):
            split = line.split()
            links = list(map(int, split[3:]))
            self.maxLinks = max(len(links), self.maxLinks)
            self.nodes.append(Node(index + 1, links))


class Model:
    def __init__(self):
        self.Graph = Graph()
        self.maxLinks = self.Graph.maxLinks
        self.currentSolution = self.generate_random_solution()
        self.cuts = self.calculate_cuts()

    def calculate_cuts(self):
        cut = 0
        for node in self.Graph.nodes:
            own_value = self.currentSolution[node.index - 1]
            for link in node.links:
                if own_value != self.currentSolution[link - 1]:
                    cut += 1
        return int(cut / 2)

    def generate_random_solution(self):
        node_amount = len(self.Graph.nodes)
        solution = np.repeat(0, node_amount)
        solution[int(node_amount / 2):] = np.repeat(1, int(node_amount / 2))
        np.random.seed(1)
        np.random.shuffle(solution)
        return solution

    def generate_buckets(self):
        left = [NodeDoubleLinkedList() for _ in range(self.maxLinks * 2 + 1)]
        right = [NodeDoubleLinkedList() for _ in range(self.maxLinks * 2 + 1)]
        return left, right

    def calculate_gain_node(self, node):
        gain = 0
        own_value = self.currentSolution[node.index - 1]
        for link in node.links:
            if own_value == self.currentSolution[link - 1]:
                gain -= 1
            else:
                gain += 1
        node.gain = gain

    def calculate_gains_graph(self, left, right):
        for node in self.Graph.nodes:
            own_value = self.currentSolution[node.index - 1]
            self.calculate_gain_node(node)
            if own_value == 0:
                left[node.gain].append(node)
            else:
                right[node.gain].append(node)
        return left, right

    def swap_best(self, use_left, left_bucket, right_bucket):
        if use_left:
            current_bucket = left_bucket
        else:
            current_bucket = right_bucket

        for i in range(self.maxLinks, -self.maxLinks - 1, -1):
            if current_bucket[i].count > 0:
                selected_node = current_bucket[i].pop()
                # print(f"Selected Node Index: {selected_node.index}   Gain:{selected_node.gain}")
                self.currentSolution[selected_node.index - 1] = 1 - self.currentSolution[selected_node.index - 1]

                # Update neighbours
                for neighbour in [self.Graph.nodes[link - 1] for link in selected_node.links]:
                    # Remove from the bucket it is in
                    if neighbour.is_in_bucket:
                        if self.currentSolution[neighbour.index - 1] == 0:
                            left_bucket[neighbour.gain].remove_node(neighbour)
                        else:
                            right_bucket[neighbour.gain].remove_node(neighbour)

                        # Recalculate gain and add to the correct bucket
                        self.calculate_gain_node(neighbour)
                        if self.currentSolution[neighbour.index - 1] == 0:
                            left_bucket[neighbour.gain].append(neighbour)
                        else:
                            right_bucket[neighbour.gain].append(neighbour)
                self.cuts -= selected_node.gain
                break

    def is_left_bucket_best(self, left_bucket, right_bucket):
        for i in range(self.maxLinks, -self.maxLinks - 1, -1):
            if len(left_bucket[i]) > 0:
                return True
            if len(right_bucket[i]) > 0:
                return False

    def print_partition(self):
        l, r = [], []
        for i, v in enumerate(self.currentSolution):
            if v == 0:
                l.append(i + 1)
            else:
                r.append(i + 1)
        print(f"left: {l}  |   right: {r}")
        print(f"Cuts: {self.cuts}")

    def fm_pass(self):
        # initialize partition and buckets
        left, right = self.generate_buckets()
        left_bucket, right_bucket = self.calculate_gains_graph(left, right)
        use_left = True  ## self.is_left_bucket_best(left_bucket, right_bucket)
        bestSolution = copy.deepcopy(self.currentSolution)  # Store a copy of the best solution
        print(f"Initial: {bestSolution}", f"Cuts: {self.cuts}")
        self.print_partition()
        bestScore = self.cuts

        # Run
        for i in range(len(self.Graph.nodes)):
            print(f"\nIteration: {i + 1}")
            # Todo: keep track of best solution.
            self.swap_best(use_left, left_bucket, right_bucket)
            if self.cuts < bestScore and i % 2 == 1:        # Only look for better score every two swaps,
                print(f"best solution: {bestSolution}")     # because we need an equal amount of nodes in each group.
                bestSolution = copy.deepcopy(self.currentSolution)
                bestScore = self.cuts
            use_left = not use_left
            self.print_partition()

        print(f"\nBest: {bestSolution}  Cuts: {bestScore}")
        print(f"Last: {self.currentSolution}  Cuts: {self.cuts}")


x = Model()
x.fm_pass()
# cProfile.run('x.fm_pass()')

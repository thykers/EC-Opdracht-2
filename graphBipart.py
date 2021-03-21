import numpy as np
# import list
import copy
import cProfile
import random
import time
debug = False
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
    global debug
    def __init__(self, gain):
        self.head = None
        self.tail = None
        self.count = 0
        self.gain = gain

    def append(self, node):
        if(debug and node.gain != self.gain):
            print(f"Adding node with gain {node.gain}",f"to bucket {self.gain}")
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
        if debug:
            if node.prev != None:
                if node.prev.gain != node.gain:
                    print("Wrong append")
            if node.next != None:
                if node.next.gain != node.gain:
                    print("Wrong append")
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
        if(self.head == None):
            return False
        current = self.head
        while True and current != None:
            if(current.index == node.index):
                return True
            if current.next == None:
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
    global debug
    fmpass_count = 0
    fmpass_limit = 500

    def __init__(self):
        self.Graph = Graph()
        self.maxLinks = self.Graph.maxLinks
        self.reset()

    def reset(self):
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
        #np.random.seed(1)
        np.random.shuffle(solution)
        return solution

    def get_gain(self, i):
        return [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1][i]

    def generate_buckets(self):
        left = [NodeDoubleLinkedList(self.get_gain(i)) for i in range(self.maxLinks * 2 + 1)]
        right = [NodeDoubleLinkedList(self.get_gain(i)) for i in range(self.maxLinks * 2 + 1)]
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
                            if debug:
                                if not (left_bucket[neighbour.gain].exists(neighbour)):
                                    for i in range(self.maxLinks, -self.maxLinks - 1, -1):
                                        if left_bucket[i].exists(neighbour):
                                            print(f"Found neighbour in bucket {i}", f"instead of {neighbour.gain}")
                            left_bucket[neighbour.gain].remove_node(neighbour)
                        else:
                            if debug:
                                if not (right_bucket[neighbour.gain].exists(neighbour)):
                                    for i in range(self.maxLinks, -self.maxLinks - 1, -1):
                                        if right_bucket[i].exists(neighbour):
                                            print(f"Found neighbour in bucket {i}", f"instead of {neighbour.gain}")
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
            if left_bucket[i].count > 0:
                return True
            if right_bucket[i].count > 0:
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
        #print(f"Initial: {bestSolution}", f"Cuts: {self.cuts}")
        #self.print_partition()
        bestScore = self.cuts

        # Run
        for i in range(len(self.Graph.nodes)):
            #print(f"\nIteration: {i + 1}")
            # Todo: keep track of best solution.
            self.swap_best(use_left, left_bucket, right_bucket)
            if self.cuts < bestScore and i % 2 == 1:        # Only look for better score every two swaps,
                #print(f"best solution: {bestSolution}")     # because we need an equal amount of nodes in each group.
                bestSolution = copy.deepcopy(self.currentSolution)
                bestScore = self.cuts
            use_left = not use_left
            #self.print_partition()

        #print(f"\nBest: {bestSolution}  Cuts: {bestScore}")
        #print(f"Last: {self.currentSolution}  Cuts: {self.cuts}")
        self.fmpass_count += 1
        return bestSolution, bestScore
    
    def MLS(self):
        MLS_solution = copy.deepcopy(self.currentSolution)
        MLS_score = self.cuts
        while self.fmpass_count < self.fmpass_limit:
            
            current_run_best = copy.deepcopy(self.currentSolution)
            current_run_best_cuts = self.cuts
            while True:
                newsolution, newscore = self.fm_pass()
                if np.array_equal(newsolution, current_run_best):
                    #print(f"converged with {newscore} cuts")
                    break
                if newscore < current_run_best_cuts:
                    current_run_best = copy.deepcopy(newsolution)
                    current_run_best_cuts = newscore
                
            if current_run_best_cuts < MLS_score:
                MLS_solution = copy.deepcopy(current_run_best)
                MLS_score = current_run_best_cuts
            self.reset()
        print(self.fmpass_count)
        print(MLS_solution, MLS_score)
        return MLS_solution, MLS_score
            


x = Model()
timer = time.time()
x.MLS()
print(f"Elapsed: {time.time() - timer}")



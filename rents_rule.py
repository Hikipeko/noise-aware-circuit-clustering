from collections import defaultdict
import numpy as np
from clusterparser import ClusterParser
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class RentsRuleCalculator(object):
    def __init__(self, cp: ClusterParser):
        self.cp = cp
        self.x_pos = cp.data[:, -2]
        self.y_pos = cp.data[:, -1]
        self.W = max(self.x_pos)
        self.H = max(self.y_pos)
        self.__cluster_rents__()
        self.__block_rents__()
        self.__icn_rents__()

    def __cluster_rents__(self):
        self.cluster_num_pins = [0] * (len(self.cp.unique_labels) - 1)
        for net in self.cp.net_nodes:
            clusters = set()
            for pin in net:
                node = pin[0]
                if node < len(self.cp.labels) and self.cp.labels[node] != -1:
                    clusters.add(self.cp.labels[node])
                else:
                    clusters.add(node)
            # ignore nets inside a cluster
            if len(clusters) == 1:
                continue
            for c in clusters:
                if c in self.cp.unique_labels and c != -1:
                    self.cluster_num_pins[c] += 1

        self.cluster_size = [0] * (len(self.cp.unique_labels) - 1)
        for i in self.cp.labels:
            if i == -1:
                continue
            self.cluster_size[i] += 1

    def __block_rents__(self):
        block_num_nodes = defaultdict(int)
        block_num_pins = defaultdict(int)

        bins = [4, 8, 16]
        for b in bins:
            for x, y in zip(self.x_pos, self.y_pos):
                i = int(x * b / self.W)
                j = int(y * b / self.H)
                block_num_nodes[(i, j, b)] += 1

            for net in self.cp.net_nodes:
                blocks = set()
                for pin in net:
                    node = pin[0]
                    if node >= len(self.x_pos):
                        continue
                    i = int(self.x_pos[node] * b / self.W)
                    j = int(self.y_pos[node] * b / self.H)
                    blocks.add((i, j, b))
                # ignore nets inside a cluster
                if len(blocks) == 1:
                    continue
                for block in blocks:
                    block_num_pins[block] += 1

        self.block_num_nodes_list = []
        self.block_num_pins_list = []

        for block in block_num_nodes:
            if block_num_nodes[block] > 3:
                self.block_num_nodes_list.append(block_num_nodes[block])
                self.block_num_pins_list.append(block_num_pins[block])

    def __icn_rents__(self):
        bins = [4, 8, 16]
        self.intercell_num_nodes_list, self.intercell_num_pins_list = (
            self.__power_law__(bins, -1)
        )

    def __power_law__(self, bins, target_label):
        intercell_num_nodes = defaultdict(int)
        intercell_num_pins = defaultdict(int)
        for b in bins:
            for node in range(len(self.x_pos)):
                i = int(self.x_pos[node] * b / self.W)
                j = int(self.y_pos[node] * b / self.H)
                if self.cp.labels[node] == target_label:
                    intercell_num_nodes[(i, j, b)] += 1

            for net in self.cp.net_nodes:
                # (has_target_cell, has_other_cell)
                blocks_intercell_cnt = defaultdict(lambda: [False, False])
                for pin in net:
                    node = pin[0]
                    if node >= len(self.x_pos):
                        continue
                    i = int(self.x_pos[node] * b / self.W)
                    j = int(self.y_pos[node] * b / self.H)
                    if self.cp.labels[node] == target_label:
                        blocks_intercell_cnt[(i, j, b)][0] = True
                    else:
                        blocks_intercell_cnt[(i, j, b)][1] = True
                # ignore nets inside a cluster
                if (
                    len(blocks_intercell_cnt) == 1
                    and blocks_intercell_cnt[(i, j, b)][0]
                    and blocks_intercell_cnt[(i, j, b)][1]
                ):
                    block = next(iter(blocks_intercell_cnt))
                    intercell_num_pins[block] += 1
                for block in blocks_intercell_cnt:
                    if blocks_intercell_cnt[(i, j, b)][0]:
                        intercell_num_pins[block] += 1

        intercell_num_nodes_list = []
        intercell_num_pins_list = []

        for block in intercell_num_nodes:
            if intercell_num_nodes[block] > 3:
                if intercell_num_pins[block] == 0:
                    continue
                intercell_num_nodes_list.append(intercell_num_nodes[block])
                intercell_num_pins_list.append(intercell_num_pins[block])

        return intercell_num_nodes_list, intercell_num_pins_list

    def approximate(self):
        # block
        block_log_gates = np.log(self.block_num_nodes_list).reshape(-1, 1)
        block_log_pins = np.log(self.block_num_pins_list).reshape(-1, 1)
        model = LinearRegression()
        model.fit(block_log_gates, block_log_pins)
        self.block_p = model.coef_[0][0]
        self.block_k = np.exp(model.intercept_[0])
        print(f"Approximated Rent's exponent (block p): {self.block_p:.2f}")
        print(f"Approximated Rent's coefficient (block dk): {self.block_k:.2f}")
        block_gates_range = np.linspace(
            min(self.block_num_nodes_list), max(self.block_num_nodes_list), 100
        )
        block_approximated_pins = self.block_k * block_gates_range**self.block_p

        # cluster
        cluster_log_gates = np.log(self.cluster_size).reshape(-1, 1)
        cluster_log_pins = np.log(self.cluster_num_pins).reshape(-1, 1)
        model.fit(cluster_log_gates, cluster_log_pins)
        self.cluster_p = model.coef_[0][0]
        self.cluster_k = np.exp(model.intercept_[0])
        print(f"Approximated Rent's exponent (cluster p): {self.cluster_p:.2f}")
        print(f"Approximated Rent's coefficient (cluster k): {self.cluster_k:.2f}")
        cluster_gates_range = np.linspace(
            0.1 * min(self.cluster_size), 2 * max(self.cluster_size), 100
        )
        cluster_approximated_pins = self.cluster_k * cluster_gates_range**self.cluster_p

        # ICN
        inter_log_gates = np.log(self.intercell_num_nodes_list).reshape(-1, 1)
        inter_log_pins = np.log(self.intercell_num_pins_list).reshape(-1, 1)
        model.fit(inter_log_gates, inter_log_pins)
        self.inter_p = model.coef_[0][0]
        self.inter_k = np.exp(model.intercept_[0])
        print(f"Approximated Rent's exponent (p): {self.inter_p:.2f}")
        print(f"Approximated Rent's coefficient (k): {self.inter_k:.2f}")
        inter_gates_range = np.linspace(
            min(self.intercell_num_nodes_list),
            1.5 * max(self.intercell_num_nodes_list),
            100,
        )
        inter_approximated_pins = self.inter_k * inter_gates_range**self.inter_p

        # Plot
        plt.figure(figsize=(10, 8), dpi=100)
        plt.scatter(
            self.block_num_nodes_list,
            self.block_num_pins_list,
            label="Whole Circuit",
            s=13,
            color="blue",
        )
        plt.scatter(
            self.cluster_size,
            self.cluster_num_pins,
            label="Modular Cluster",
            s=18,
            color="red",
        )
        plt.scatter(
            self.intercell_num_nodes_list,
            self.intercell_num_pins_list,
            label="Inter-Cluster Network",
            s=13,
            color="green",
        )
        plt.plot(
            block_gates_range,
            block_approximated_pins,
            color="blue",
            label=f"Whole Circuit Rent's rule",
        )
        plt.plot(
            cluster_gates_range,
            cluster_approximated_pins,
            color="red",
            label=f"Modular Cluster Rent's rule",
        )
        plt.plot(
            inter_gates_range,
            inter_approximated_pins,
            color="green",
            label=f"Inter-Cluster Network Rent's rule",
        )

        plt.xlabel("Number of cells", fontsize=14)
        plt.ylabel("Number of pins", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.title('Scatter Plot of Cluster Size vs. Cluster Number of Pins')
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(fontsize=14)

    def intra_rents(self):
        bins2 = [8, 16, 32, 64, 128]
        cluster_nodes = []
        cluster_pins = []

        for i in range(len(self.cp.unique_labels)):
            cluster_nodes_i, cluster_pins_i = self.__power_law__(bins2, i)
            cluster_nodes.append(cluster_nodes_i)
            cluster_pins.append(cluster_pins_i)
        intra_ps = []
        intra_ks = []

        for i in range(len(cluster_nodes)):
            if min(cluster_pins[i]) <= 1:
                continue
            intra_log_gates = np.log(cluster_nodes[i]).reshape(-1, 1)
            intra_log_pins = np.log(cluster_pins[i]).reshape(-1, 1)
            model = LinearRegression()
            model.fit(intra_log_gates, intra_log_pins)
            intra_p = model.coef_[0][0]
            intra_k = np.exp(model.intercept_[0])
            intra_ps.append(intra_p)
            intra_ks.append(intra_k)

        # plt.figure(figsize=(10, 8), dpi=200)

        # plt.scatter(intra_ps, intra_ks, label=f"Intra-Cluster Network Rent's rule", s=8, color='#FF6000')
        # plt.scatter([0.56], [13.48], label=f"Whole Circuit Rent's rule", s=50, color='blue')
        # plt.scatter([0.92], [3.76], label=f"Inter-Cluster Network Rent's rule", s=50, color='green')
        # plt.scatter([0.21], [71.44], label="Modular Cluster Rent's rule", s=50, color='purple')
        # plt.axvline(x=0.56, color='blue', linestyle='--')
        # plt.axvline(x=0.92, color='green', linestyle='--')
        # plt.axvline(x=0.21, color='purple', linestyle='--')

        # plt.xlabel('p', fontsize=14)
        # plt.ylabel('K', fontsize=14)

        # plt.xticks(fontsize=12)
        # plt.yticks(fontsize=12)
        # plt.legend(fontsize=13)

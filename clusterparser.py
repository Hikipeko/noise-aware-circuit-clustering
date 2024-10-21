import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from typing import List
import os

STD_CELL = 0
MACRO = 1
TERMINAL = 2
CLUSTER = 3
NOISY = -1


class ClusterParser(object):
    def __init__(
        self, node_file: str, net_file: str = None, macro_criteria: int = None
    ):
        """Initialize with nodes and nets information. Net file is not necessacry for clustering.

        Args:
            node_file (str): .nodes file
            net_file (str, optional): .nets file. Defaults to None. Not necessary for clustering. Only used for Rent's rule and small world analysis.
            macro_criteria (str, optional): if set, cells having height/width >= macro_criteria will be macros
        """
        self.__load_nodes(node_file, macro_criteria)
        if net_file:
            self.__load_nets(net_file)

    def __comment_line(self, line):
        if (
            line.startswith("UCLA")
            or line.startswith("#")
            or line.startswith("NumNodes")
            or line.startswith("NumTerminals")
            or line.startswith("NumNets")
            or line.startswith("NumPins")
            or not line
            or line == "\n"
        ):
            return True

    def __load_nodes(self, filename: str, macro_criteria: int = None):
        self.node_names = []
        self.node_size_x = []
        self.node_size_y = []
        self.node_type = []
        self.node_name2id_map = {}
        with open(filename, "r") as file:
            lines = file.readlines()

        for line in lines:
            if self.__comment_line(line):
                continue
            tokens = line.split()
            if len(tokens) < 3:
                continue
            node_name = tokens[0]
            width = int(float(tokens[1]))
            height = int(float(tokens[2]))

            self.node_names.append(node_name)
            self.node_size_x.append(width)
            self.node_size_y.append(height)
            self.node_name2id_map[node_name] = len(self.node_names) - 1
            # cell type
            if len(tokens) == 4:
                self.node_type.append(TERMINAL)
            elif macro_criteria is not None and (
                width >= macro_criteria or height >= macro_criteria
            ):
                self.node_type.append(MACRO)
            else:
                self.node_type.append(STD_CELL)
        self.num_std_cells = self.node_type.count(STD_CELL)
        self.num_macros = self.node_type.count(MACRO)
        self.num_terminals = self.node_type.count(TERMINAL)

    def __load_nets(self, filename: str):
        # Read in .nets file
        current_net_name = None
        current_net_nodes = []
        self.net_names = []
        self.net_name2id_map = {}
        self.net_nodes = []
        with open(filename, "r") as file:
            lines = file.readlines()

        for line in lines:
            if self.__comment_line(line):
                continue
            # Check if the line indicates a new net
            if line.startswith("NetDegree"):
                # If there's a current net being processed, save it before starting the new one
                if current_net_name is not None:
                    self.net_names.append(current_net_name)
                    self.net_name2id_map[current_net_name] = len(self.net_names) - 1
                    self.net_nodes.append(current_net_nodes)

                parts = line.split()
                current_net_name = parts[-1]
                # The net name, corrected to the last part of the line
                current_net_nodes = []  # Reset for the new net
                continue  # Move to the next iteration

            # Corrected process node lines to handle parts length dynamically
            parts = line.strip().split()
            if len(parts) >= 5:
                node_name, direction, x_offset, y_offset = (
                    parts[0],
                    parts[1],
                    float(parts[3]),
                    float(parts[4]),
                )
                node_id = self.node_name2id_map.get(node_name, None)
                if node_id is not None:
                    current_net_nodes.append((node_id, x_offset, y_offset, direction))

        # Add the last net if there was one
        if current_net_name is not None:
            self.net_names.append(current_net_name)
            self.net_name2id_map[current_net_name] = len(self.net_names) - 1
            self.net_nodes.append(current_net_nodes)

    def load_terminals(self, filename: str, is_terminal=None):
        """Load the position of terminals. (deprecated)

        In our implementation, we distinguish between macro cells and terminals. Terminals have fixed position, while macro cells will be placed in macro placement stage.

        Args:
            filename (str): .pl file
            is_terminal ((int, int) -> bool): a function judging whether a cell is a terminal
        """
        self.terminal_pos = {}  # idx -> (x, y)
        with open(filename, "r") as file:
            input_data = file.readlines()
        for line in input_data:
            if not line.endswith("/FIXED\n"):
                continue
            tokens = line.strip().split()
            node_name, x, y = tokens[0], int(tokens[1]), int(tokens[2])
            if is_terminal(x, y):
                idx = self.node_name2id_map[node_name]
                assert self.node_type[idx] != STD_CELL
                self.node_type[idx] = TERMINAL
                self.terminal_pos[idx] = (x, y)

    def load_positions(self, input_files: list[str]):
        """Initialize positions from placement snapshots. Each line looks like
        cell_name x_pos y_pos : dir

        Args:
            input_files: input files of placement snapshots
        """
        data_points = {}

        for file_path in input_files:
            with open(file_path, "r") as file:
                input_data = file.readlines()
            for line in input_data:
                if self.__comment_line(line):
                    continue
                tokens = line.strip().split()
                node_name, x, y = tokens[0], float(tokens[1]), float(tokens[2])
                idx = self.node_name2id_map[node_name]
                if idx not in data_points:
                    data_points[idx] = []
                data_points[idx].extend([float(x), float(y)])

        # Convert data points to NumPy array
        self.data = np.array(list(data_points.values()))
        self.num_snapshots = len(input_files)

    def clustering(
        self, min_size_list: List[int], min_samples: int = 1, epsilon: float = 0.0
    ):
        """Perform HDBSCAN clustering.

        Args:
            min_size_list (List[int]): List of min_cluster_size for each HDBSCAN round
            min_samples (int, optional): HDBSCAN min_samples. Defaults to 1.
            epsilon (float, optional): HDBSCAN epsilon. Defaults to 0.0.
        """
        max_label = -1

        # first round
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_size_list[0],
            cluster_selection_epsilon=epsilon,
            min_samples=min_samples,  # min_size_list[0]
        )
        self.labels = clusterer.fit_predict(self.data)
        self.unique_labels = np.unique(self.labels)
        self.noise_points = self.data[self.labels == -1]

        # following rounds
        for min_size in min_size_list[1:]:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_size,
                cluster_selection_epsilon=epsilon,
                min_samples=min_samples,  # min_size
            )
            new_labels = clusterer.fit_predict(self.noise_points)

            # Adjust the new_labels to ensure unique cluster IDs
            max_label = max(self.labels.max(), max_label)
            adjusted_labels = np.where(new_labels == -1, -1, new_labels + max_label + 1)
            noise_indices = np.where(self.labels == -1)[0]
            self.labels[noise_indices] = adjusted_labels

            # Update the max_label

            self.unique_labels = np.unique(self.labels)
            self.noise_points = self.data[self.labels == -1]

        self.cluster_centers = {}
        # Iterate over unique labels
        for label in self.unique_labels:
            if label != -1:
                # Extract points belonging to the current cluster label
                points_in_cluster = self.data[self.labels == label]
                self.cluster_centers[label] = np.mean(points_in_cluster, axis=0)

    def save_clustering(self, cluster_file: str, output_file: str = None):
        """Save the generated clustering data.

        Args:
            cluster_file (str): csv file storing only the labels
            output_file (str, optional): human readable output file describing the label of each node. Defaults to None.
        """
        np.savetxt(cluster_file, self.labels, fmt="%s")
        if output_file:
            with open(output_file, 'w') as f:
                for node_name, label in zip(self.node_names, self.labels):
                    f.write(f"{node_name}\t{label}\n")
            

    def load_clustering(self, filename: str):
        """Load the generated clustering data.

        Args:
            filename (str): cluster.csv file
        """
        self.labels = np.loadtxt(filename, dtype=int)
        self.unique_labels = np.unique(self.labels)
        self.noise_points = self.data[self.labels == -1]

    def plot_snapshot(self, i=None, labels=None, noise=True, macro=True, output_file=None):
        """Plot the placement snapshot with clustering to see how clustering evolve with DREAMPlace placement.

        Args:
            i (int): the snapshot id, default to last snapshot
            labels (list[int], optional): labels of clusters to be plotted. Defaults to plot all of the clusters.
            noise (bool, optional): whether plot noise points. Defaults to True.
            macro (bool, optional): whether plot macros. Defaults to False.
            output_file (str, optional): store to output_file if not None.
        """
        if not i:
            i = self.num_snapshots - 1
        if not labels:
            labels = self.unique_labels
            
        plt.figure(figsize=(16, 16), dpi=200)
        if macro:
            for j in range(len(self.node_names)):
                if self.node_type[j] == MACRO:
                    plt.gca().add_patch(
                        plt.Rectangle(
                            (self.data[j, i * 2], self.data[j, i * 2 + 1]),
                            self.node_size_x[j],
                            self.node_size_y[j],
                            fill=True,
                            facecolor="grey",
                            edgecolor="black",
                            alpha=0.15,
                        )
                    )
        if noise:
            plt.scatter(
                self.noise_points[:, i * 2],
                self.noise_points[:, i * 2 + 1],
                color="black",
                s=0.2,
                label=None,
                # marker='x'
            )
        # Clusters
        for label in labels:
            if label != -1:
                cluster_points = self.data[self.labels == label]
                plt.scatter(
                    cluster_points[:, i * 2],
                    cluster_points[:, i * 2 + 1],
                    s=0.1,
                    label=None,
                )
        # if hasattr(self, 'cluster_centers'):
        #     for label, center in self.cluster_centers.items():
        #         if label != -1 and label in l:
        #             plt.scatter(center[i*2], center[i*2+1], color='red', marker='x', s=10, label=None)
        lim = np.max((self.data))
        plt.xlim(lim * 0.00, lim * 1.0)
        plt.ylim(lim * 0.00, lim * 1.0)
        # plt.legend()
        if not output_file:
            plt.show()
        else:
            plt.savefig(output_file)

    def parse_softmacros(self, labels=None):
        """Parse standard cells into clusters based on their labels

        Args:
            scale (float, optional): Scale of macro expansion. Defaults to 1.0.
        """
        assert self.labels is not None
        if not labels:
            labels = list(range(max(self.labels)))

        # Parsing nodes
        self.new_node_names = []  # Stores the names of new softmacro nodes
        self.new_node_size_x = []
        self.new_node_size_y = []
        self.new_node_type = []
        self.new_node_name2id_map = {}

        # Generate a soft macro for each cluster
        for label in labels:
            name = f"c{label}"  # Name the new softmacros as c0, c1, etc.
            self.new_node_names.append(name)
            self.new_node_name2id_map[name] = len(self.new_node_names) - 1
            # Compute total area
            valid_indices = np.where(self.labels == label)[0]
            cluster_area = 0
            for i in valid_indices:
                cluster_area += self.node_size_x[i] * self.node_size_y[i]
            w = int(cluster_area**0.5)
            self.new_node_size_x.append(w)
            self.new_node_size_y.append(w)
            self.new_node_type.append(CLUSTER)

        # Add hard macros, terminals, and stdcells not belonging to good clusters
        for i in range(len(self.node_names)):
            if (
                self.node_type[i] == MACRO
                or self.node_type[i] == TERMINAL
                or self.labels[i] not in labels
            ):
                name = self.node_names[i]
                self.new_node_names.append(name)
                self.new_node_name2id_map[name] = len(self.new_node_names) - 1
                self.new_node_size_x.append(int(self.node_size_x[i]))
                self.new_node_size_y.append(int(self.node_size_y[i]))
                self.new_node_type.append(self.node_type[i])

        # Parsing nets
        self.new_nets = []
        self.new_nets_weight = []
        for net in self.net_nodes:
            new_net = []
            # nodes containing pin for this net, deduplicate pins
            node_set = set()
            for pin in net:
                cell_id, x_offset, y_offset = pin
                if (
                    self.node_type[cell_id] != STD_CELL
                    or self.labels[cell_id] not in labels
                ):
                    # If macro / stdcell not belonging to good clusters, remain the pin offset
                    node_name = self.node_names[cell_id]
                    cell_new_id = self.new_node_name2id_map[node_name]
                    new_net.append((cell_new_id, x_offset, y_offset))
                    node_set.add(cell_new_id)
                else:
                    # If stdcell, assume the pin is the center of the cluster
                    cluster = self.labels[pin[0]]
                    if cluster in node_set:
                        continue
                    cluster_id = self.new_node_name2id_map[f"c{label}"]
                    new_net.append((cluster_id, 0.0, 0.0))
                    node_set.add(cluster)
            if len(new_net) > 1:
                self.new_nets.append(new_net)

    def is_noisy(self, id):
        return self.labels[id] == -1

    def produce_btree_input(self, nodefile, netfile, outline_x, outline_y):
        # Output .node file for B-Tree
        with open(nodefile, "w") as file:
            file.write(f"Outline: {outline_x} {outline_y}\n")
            file.write(f"NumBlocks: {len(self.new_node_names) - self.num_terminal}\n")
            file.write(f"NumTerminals: {self.num_terminal}\n\n")

            data = zip(
                self.new_node_names,
                self.new_node_size_x,
                self.new_node_size_y,
                self.new_node_type,
            )
            for name, width, height, type in data:
                if type == CLUSTER or type == MACRO:
                    file.write(f"{name}\t{width}\t{height}\n")
            file.write("\n")

            # print terminals
            for i, name in enumerate(self.new_node_names):
                if self.new_node_type[i] == TERMINAL:
                    terminal_id = self.node_name2id_map[name]
                    x, y = self.terminal_pos[terminal_id]
                    file.write(f"{name}\tterminal\t{x}\t{y}\n")

        # Output .net file for B-Tree
        with open(netfile, "w") as file:
            file.write(f"NumNets: {len(self.new_nets)}\n")
            for i, net in enumerate(self.new_nets):
                file.write(f"NetDegree: {len(net)}\n")
                for pin in net:
                    node_name = self.new_node_names[pin[0]]
                    file.write(f"{node_name}\n")

    def load_btree_output(self, plfile, outfile):
        with open(plfile, "r") as in_file:
            lines = in_file.readlines()
        with open(outfile, "w") as out_file:
            out_file.write("UCLA pl 1.0\n")
            # Nodes
            for line in lines:
                tokens = line.split()
                if len(tokens) == 5:
                    node_name = tokens[0]
                    x = int(tokens[1])
                    y = int(tokens[2])
                out_file.write(f"{node_name}\t{x}\t{y}\t: N\n")
            # Terminals
            for name, type in zip(self.new_node_names, self.new_node_type):
                if type == TERMINAL:
                    terminal_id = self.node_name2id_map[name]
                    x, y = self.terminal_pos[terminal_id]
                    out_file.write(f"{name}\t{x}\t{y}\t: N /FIXED\n")

    def produce_dreamplace_input(self, nodefile, netfile, plfile):
        # Output .node file
        with open(nodefile, "w") as file:
            file.write("UCLA nodes 1.0\n\n")
            file.write(f"NumNodes :\t{len(self.new_node_names)}\n")
            file.write(f"NumTerminals :\t{self.num_terminals}\n")
            # Nodes
            for name, width, height, type in zip(
                self.new_node_names,
                self.new_node_size_x,
                self.new_node_size_y,
                self.new_node_type,
            ):
                if type == TERMINAL:
                    file.write(f"\t{name}\t{width}\t{height}\tterminal\n")
                else:
                    file.write(f"\t{name}\t{width}\t{height}\n")

        # Output .net file
        total_pins = sum(len(net) for net in self.new_nets)
        with open(netfile, "w") as file:
            file.write("UCLA nets 1.0\n\n")
            file.write(f"NumNets : {len(self.new_nets)}\n")
            file.write(f"NumPins : {total_pins}\n\n")

            for i, net in enumerate(self.new_nets):
                file.write(f"NetDegree : {len(net)}   n{i}\n")
                for pin in net:
                    node_name = self.new_node_names[pin[0]]
                    direction = "I"
                    x_offset, y_offset = pin[1], pin[2]
                    file.write(f"\t{node_name}\t{direction} : {x_offset}\t{y_offset}\n")

        # Output .pl file
        with open(plfile, "w") as file:
            for name, width, height, type in zip(
                self.new_node_names,
                self.new_node_size_x,
                self.new_node_size_y,
                self.new_node_type,
            ):
                if type == TERMINAL:
                    tid = self.node_name2id_map[name]
                    x_pos, y_pos = self.terminal_pos[tid]
                    # o211411 22 4795 : N /FIXED
                    file.write(f"{name} {x_pos} {y_pos} : N /FIXED\n")
                else:
                    file.write(f"{name} 0 0 : N\n")

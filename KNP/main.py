import random

import networkx
import numpy as np
import networkx as nx
import matplotlib.pyplot as plot


def get_clusters_color():
    return random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)


def draw(graph, clusters):
    graph_view = nx.Graph(strict=False)

    vertices_count = len(graph)
    colors = []
    for cluster in clusters:
        color = get_clusters_color()
        for point in cluster:
            graph_view.add_node(point)
            colors.append(color)
    for i in range(vertices_count):
        for j in range(i + 1, vertices_count):
            if graph[i][j] > 0:
                graph_view.add_edge(i, j, weight="{:.1f}".format(graph[i][j]))

    graph_view_position = networkx.random_layout(graph_view)
    networkx.draw_networkx(graph_view, graph_view_position, node_color=colors, with_labels=True)
    edge_labels = networkx.get_edge_attributes(graph_view, "weight")
    networkx.draw_networkx_edge_labels(graph_view, pos=graph_view_position, edge_labels=edge_labels)
    plot.show()


def generate_graph(min_nodes_count, max_nodes_count):
    points_count = random.randint(min_nodes_count, max_nodes_count)
    points_coordinates = []
    for i in range(points_count):
        points_coordinates.append([random.randint(0, 100), random.randint(0, 100)])

    graph = get_tree(points_coordinates)

    edges_count = random.randint(points_count, points_count * (points_count - 1) // 2 - 1)
    edged_count_to_add = edges_count - (points_count - 1)
    for i in range(edged_count_to_add):
        edge_is_not_added = True
        while edge_is_not_added:
            first_node_index = random.randint(0, points_count - 1)
            second_node_index = random.randint(0, points_count - 1)
            while first_node_index == second_node_index:
                second_node_index = random.randint(0, points_count - 1)

            if graph[first_node_index][second_node_index] == 0:
                add_edge(graph, points_coordinates, first_node_index, second_node_index)
                edge_is_not_added = False

    return graph, points_coordinates


def get_distance(first_point, second_point):
    return np.sqrt((first_point[0] - second_point[0]) ** 2 +
                   (first_point[1] - second_point[1]) ** 2)


def get_tree(points_coordinates):
    points_count = len(points_coordinates)
    tree = [[0.0 for _ in range(points_count)] for _ in range(points_count)]
    edge_count = points_count - 1

    first_node_index = random.randint(0, points_count)
    second_node_index = random.randint(0, points_count)
    while first_node_index == second_node_index:
        second_node_index = random.randint(0, points_count)
    add_edge(tree, points_coordinates, first_node_index, second_node_index)

    connected_graph_points = {first_node_index, second_node_index}

    for i in range(edge_count - 1):
        first_node_index = random.choice(list(connected_graph_points))
        second_node_index = random.randint(0, points_count - 1)
        while first_node_index == second_node_index or second_node_index in connected_graph_points:
            second_node_index = random.randint(0, points_count - 1)

        add_edge(tree, points_coordinates, first_node_index, second_node_index)
        connected_graph_points.add(second_node_index)

    return tree


def add_edge(graph, points_coordinates, first_node_index, second_node_index):
    distance = get_distance(points_coordinates[first_node_index], points_coordinates[second_node_index])
    graph[first_node_index][second_node_index] = distance
    graph[second_node_index][first_node_index] = distance


def get_mst_edges(graph):
    points_count = len(graph)
    mst_edges = []
    mst_points = set()
    min_edge_length = np.sqrt(100 ** 2 * 2)
    edges_points_index = []
    for i in range(points_count):
        for j in range(i):
            current_edge_length = graph[i][j]
            if current_edge_length < min_edge_length and current_edge_length != 0:
                min_edge_length = current_edge_length
                edges_points_index = [i, j]

    mst_edges.append(edges_points_index)
    mst_points.update(edges_points_index)

    for i in range(points_count - 2):
        next_edge_to_add_to_mst = find_next_edge_to_add_to_mst(mst_points, graph)
        mst_edges.append(next_edge_to_add_to_mst)
        mst_points.update(next_edge_to_add_to_mst)

    return mst_edges


def find_next_edge_to_add_to_mst(mst_points, graph):
    points_count = len(graph)
    min_edge_length = np.sqrt(100 ** 2 * 2)
    edges_points_index = []
    for i in mst_points:
        for j in range(points_count):
            current_edge_length = graph[i][j]
            if current_edge_length < min_edge_length \
                    and current_edge_length != 0 \
                    and j not in mst_points:
                min_edge_length = current_edge_length
                edges_points_index = [i, j]

    return edges_points_index


def get_clusters(clusters_count, graph, mst_edges):
    points_count = len(graph)
    mst_edges.sort(
        key=lambda edge: graph[edge[0]][edge[1]])
    mst_edges = mst_edges[:len(mst_edges) - clusters_count + 1]
    clusters = []
    visited_points = set()
    while len(mst_edges) != 0:
        cluster = set()
        edge = mst_edges.pop()
        points_to_visit = edge.copy()
        while len(points_to_visit) != 0:
            current_point = points_to_visit.pop()
            visited_points.add(current_point)
            cluster.add(current_point)
            edge_index = 0
            while edge_index < len(mst_edges):
                current_edge = mst_edges[edge_index]
                if current_edge[0] == current_point:
                    points_to_visit.append(current_edge[1])
                    mst_edges.pop(edge_index)
                elif current_edge[1] == current_point:
                    points_to_visit.append(current_edge[0])
                    mst_edges.pop(edge_index)
                else:
                    edge_index += 1

        clusters.append(cluster)
    for i in range(points_count):
        if i not in visited_points:
            clusters.append({i})

    return clusters


def main():
    graph, _ = generate_graph(10, 15)
    edges = get_mst_edges(graph)
    clusters = get_clusters(4, graph, edges)
    draw(graph, clusters)


if __name__ == '__main__':
    main()

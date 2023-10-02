import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib.patches import Circle

epsilon = 1 / 1000000


def get_points(n=100):
    points = []
    for i in range(n):
        points.append([random.randint(0, 100),
                       random.randint(0, 100)])
    return points


def visualize_points(points, centroids, points_within_clusters):
    center, r = set_boundary(points)
    draw_boundary(center, r)
    for i in range(len(points)):
        point = points[i]
        color = points_within_clusters[i]
        if color == 0:
            plt.scatter(point[0], point[1],
                        color='black')
        if color == 1:
            plt.scatter(point[0], point[1],
                        color='green')
        if color == 2:
            plt.scatter(point[0], point[1],
                        color='blue')
        if color == 3:
            plt.scatter(point[0], point[1],
                        color='violet')
        if color == 4:
            plt.scatter(point[0], point[1],
                        color='yellow')
        if color == 5:
            plt.scatter(point[0], point[1],
                        color='pink')
        if color == 6:
            plt.scatter(point[0], point[1],
                        color='brown')
        if color == 7:
            plt.scatter(point[0], point[1],
                        color='fuchsia')
        if color == 8:
            plt.scatter(point[0], point[1],
                        color='greenyellow')
        if color == 9:
            plt.scatter(point[0], point[1],
                        color='white')

    for centroid in centroids:
        plt.scatter(centroid[0], centroid[1],
                    color='red')


def dist(pointA, pointB):
    return np.sqrt((pointA[0] - pointB[0]) ** 2 +
                   (pointA[1] - pointB[1]) ** 2)


def adjust_clusters_number(points, start_from=1, to=10):
    k = 1
    wcss = []
    for i in range(to):
        new_centroids = initialize_centroids(points, k)
        old_centroids = new_centroids
        continue_cycle = True
        t = 1

        while continue_cycle:
            print(t)
            t += 1
            points_within_clusters = nearby_point(points, new_centroids)
            visualize_points(points, new_centroids, points_within_clusters)
            plt.show()
            old_centroids = new_centroids
            new_centroids = find_new_centroids(points, new_centroids)
            continue_cycle = is_any_centroid_moved(new_centroids, old_centroids)

        clusters_points = nearby_point_as_3d_matrix(points, old_centroids)

        squared_distance_sum_between_points_and_centroid = 0
        for cluster_points, s in clusters_points:
            for point in cluster_points:
                squared_distance_sum_between_points_and_centroid = squared_distance_sum_between_points_and_centroid + \
                                                                   dist(point, old_centroids[s]) ** 2

        wcss.append(squared_distance_sum_between_points_and_centroid)

        n = len(wcss) - 2
        proper_cluster_number = 1
        for p in range(n + 1):
            first_distance_decrease_speed = abs(wcss[p] - wcss[p + 1]) / abs(wcss[p-1] - wcss[p])
            second_distance_decrease_speed = abs(wcss[p-1] - wcss[p]) / abs(wcss[p-2] - wcss[p-1])

            if first_distance_decrease_speed < second_distance_decrease_speed:
                break

            proper_cluster_number = p

        return proper_cluster_number


def set_boundary(points):
    center = [0, 0]
    for point in points:
        center[0] += point[0]
        center[1] += point[1]
    center[0] /= len(points)
    center[1] /= len(points)
    r = 0
    for point in points:
        d = dist(center, point)
        if d > r:
            r = d
    return center, r


def draw_boundary(center, r):
    plt.xlim()
    ax = plt.gca()
    circle = Circle((center[0], center[1]), r, color='gray', alpha=.3)
    ax.add_patch(circle)


def initialize_centroids(points, k):
    center, r = set_boundary(points)
    centroids = []
    for i in range(k):
        centroids.append(
            [r * np.cos(2 * np.pi * i / k) + center[0],
             r * np.sin(2 * np.pi * i / k) + center[1]])
    return centroids


def nearby_point(points, centroids):
    clusters = []
    for point in points:
        r, index = 10000000, -1
        for i in range(len(centroids)):
            d = dist(point, centroids[i])
            if r > d:
                r = d
                index = i
        clusters.append(index)
    return clusters


def nearby_point_as_3d_matrix(points, centroids):
    clusters = [[] for i in range(len(centroids))]

    for point in points:
        r, index = 10000000, -1
        for i in range(len(centroids)):
            d = dist(point, centroids[i])
            if r > d:
                r = d
                index = i

        cluster = clusters[index]
        cluster.append(point)
    return clusters


def find_new_centroids(points, centroids):
    clusters_with_points_coordinates = nearby_point_as_3d_matrix(points, centroids)
    new_centroids = []
    for i in range(len(clusters_with_points_coordinates)):
        cluster_points = clusters_with_points_coordinates[i]
        cluster_points_sum_y = 0
        cluster_points_sum_x = 0
        for point in cluster_points:
            cluster_points_sum_y += point[1]
            cluster_points_sum_x += point[0]

        new_centroid = [cluster_points_sum_x / len(cluster_points),
                        cluster_points_sum_y / len(cluster_points)]

        new_centroids.append(new_centroid)

    return new_centroids


def is_any_centroid_moved(new_centroids, old_centroids):
    delta = 0
    for new_centroid, old_centroid in zip(new_centroids, old_centroids):
        delta = dist(new_centroid, old_centroid)
        if delta > epsilon:
            return True

    return False


# def update(i, im, image_arrays):
#     im.set_array(image_arrays[i])
#     return im
#
# def create_gif():
#     # Create the figure and axes objects
#     fig, ax = plt.subplots()
#
#     # Set the initial image
#     im = ax.imshow(image_arrays[0], animated=True)
#
#     animation_fig = animation.FuncAnimation(fig, update, frames=len(image_arrays), interval=200, blit=True,
#                                             repeat_delay=10, )
#
#     # Show the animation
#     plt.show()
#
#     animation_fig.save("steps/animated_GMM.gif")


def start():
    points = get_points()
    k = adjust_clusters_number(points)
    new_centroids = initialize_centroids(points, k)
    continue_cycle = True
    t = 1

    while continue_cycle:
        print(t)
        t += 1
        points_within_clusters = nearby_point(points, new_centroids)
        visualize_points(points, new_centroids, points_within_clusters)
        plt.show()
        old_centroids = new_centroids
        new_centroids = find_new_centroids(points, new_centroids)
        result = is_any_centroid_moved(new_centroids, old_centroids)
        continue_cycle = is_any_centroid_moved(new_centroids, old_centroids)


if __name__ == '__main__':
    start()

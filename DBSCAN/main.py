import pygame
from math import dist
import random
import numpy as np


def scatter(point):
    r = 5
    n = random.randint(3,5)
    points = []
    for i in range(n):
        radius = random.randint(r, 3*r)
        angle = random.randint(0, 360)
        x = radius*np.cos(2*np.pi*angle/360)+point[0]
        y = radius*np.sin(2*np.pi*angle/360)+point[1]
        points.append((x,y))
    return points


def set_and_launch_pygame_display():
    pygame.init()
    r = 3
    screen = pygame.display.set_mode((600, 400), pygame.RESIZABLE)
    screen.fill('white')
    pygame.display.update()
    points = []
    flag = True
    mouse_button_down = False
    while flag:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                flag = False
                break
            if event.type == pygame.WINDOWEXPOSED:
                screen.fill('white')
                for point in points:
                    pygame.draw.circle(screen,
                                       color='black', center=point,
                                       radius=r)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_button_down = True
            if event.type == pygame.MOUSEBUTTONUP:
                mouse_button_down = False
            if mouse_button_down:
                coord = event.pos
                points_new = scatter(coord)

                if len(points):
                    if dist(coord, points[-1]) > 5 * r:
                        points.append(coord)

                        points.extend(points_new)
                        for point in points_new:
                            pygame.draw.circle(screen,
                                               color='black', center=point,
                                               radius=r)

                        pygame.draw.circle(screen,
                                           color='black', center=coord,
                                           radius=r)
                else:
                    points.append(coord)
                    pygame.draw.circle(screen,
                                       color='black', center=coord,
                                       radius=r)
            pygame.display.update()
    return points


def run_dbscan(points, r, neighbours_number=3, visualize_on=True):
    flagged_points = assign_flag_to_points(points, r, neighbours_number)
    clusters = group_points()
    if visualize_on:
        draw_flagged_points(points, flagged_points)
        draw_clusters(clusters)


def get_distance(first_point, second_point):
    x_fp, y_fp = first_point
    x_sp, y_sp = second_point
    distance = (abs(x_fp - x_sp) ** 2 + abs(y_fp - y_sp) ** 2) ** 0.5
    return distance


def mark_all_green_points(flagged_points, minimum_neighbours_number, r):
    for flagged_point, i in zip(flagged_points, range(len(flagged_points))):
        point, _ = flagged_point
        neighbours_number = 0
        for potential_neighbour in flagged_points:
            potential_neighbour_point, _ = potential_neighbour

            if potential_neighbour_point == point:
                continue

            distance = get_distance(point, potential_neighbour_point)

            if distance < r:
                neighbours_number += 1
                if neighbours_number == minimum_neighbours_number:
                    break

        if neighbours_number == minimum_neighbours_number:
            flagged_points[i] = (point, 'green')


def mark_all_yellow_points(flagged_points, r):
    for flagged_point, i in zip(flagged_points, range(len(flagged_points))):
        point, flag = flagged_point
        if flag == 'undefined':
            for potential_green_flagged_neighbor_point in flagged_points:
                potential_green_flagged_neighbor_point_coordinates, potential_green_flagged_neighbor_point_flag = potential_green_flagged_neighbor_point
                if potential_green_flagged_neighbor_point_flag == 'green':
                    if get_distance(point, potential_green_flagged_neighbor_point_coordinates) < r:
                        flagged_points[i] = (point, 'yellow')


def mark_all_red_points(flagged_points):
    for flagged_point, i in zip(flagged_points, range(len(flagged_points))):
        point, flag = flagged_point
        if flag == 'undefined':
            flagged_points[i] = (point, 'red')


def assign_flag_to_points(points, r, minimum_neighbours_number):
    flagged_points = [(point, "undefined") for point in points]
    mark_all_green_points(flagged_points, minimum_neighbours_number, r)
    mark_all_yellow_points(flagged_points, r)
    mark_all_red_points(flagged_points)

    return flagged_points


def group_points(flagged_points, r, minimum_neighbour_number):
    sample = set(filter(
        lambda point: point[1] == "green", flagged_points))
    clusters = []
    while len(sample) != 0:
        current_point = sample.pop()
        cluster = {current_point}
        points_to_visit = {current_point}
        clusters.append(cluster)
        while len(points_to_visit) != 0:
            point_to_visit = points_to_visit.pop()
            neighbours = find_points_neighbours(point_to_visit, flagged_points, r)
            cluster = cluster.union(neighbours)
            neighbours_that_belongs_sample = return_neighbours_that_belongs_sample(neighbours, sample)
            sample = sample - neighbours_that_belongs_sample
            point_to_visit = points_to_visit.union(neighbours_that_belongs_sample)

    return clusters


def find_points_neighbours(root_point, flagged_points, r):
    neighbours = set()
    for flagged_point in flagged_points:
        flagged_point_coordinate, flagged_point_flag = flagged_point
        if flagged_point_flag == 'green' or 'yellow':
            distance = get_distance(root_point, flagged_point_coordinate)
            if distance < r:
                neighbours.add(flagged_points)

    return neighbours


def return_neighbours_that_belongs_sample(neighbours, sample):
    neighbours_that_belongs_sample = set()
    neighbours = neighbours.copy()
    while len(neighbours) != 0:
        neighbour = neighbours.pop()
        if neighbour in sample:
            neighbours_that_belongs_sample.add(neighbour)

    return neighbours_that_belongs_sample


def draw_flagged_points(flagged_points):
    pygame.init()
    point_radius = 3
    screen = pygame.display.set_mode((600, 400), pygame.RESIZABLE)
    screen.fill('white')
    pygame.display.update()
    flag = True
    while flag:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                flag = False
                break
            if event.type == pygame.WINDOWEXPOSED:
                screen.fill('white')
                for fp in flagged_points:
                    coordinates, color = fp
                    pygame.draw.circle(screen,
                                       color=color, center=coordinates,
                                       radius=point_radius)

            pygame.display.update()


def draw_clusters(clusters):
    pygame.init()
    point_radius = 3
    screen = pygame.display.set_mode((600, 400), pygame.RESIZABLE)
    screen.fill('white')
    pygame.display.update()
    clusters_color = get_clusters_color()
    for cluster, cluster_color in zip(clusters, clusters_color):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                break
            if event.type == pygame.WINDOWEXPOSED:
                screen.fill('white')
                for cluster_point in cluster:
                    pygame.draw.circle(screen,
                                       color=cluster_color, center=cluster_point,
                                       radius=point_radius)

            pygame.display.update()


def get_clusters_color(clusters_number):
    
    for
    return NotImplementedError


def start():
    points = set_and_launch_pygame_display()
    run_dbscan(points, 20)


if __name__ == '__main__':
    start()

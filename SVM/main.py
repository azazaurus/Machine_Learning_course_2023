import random
import numpy as np
import pygame
from math import dist
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


def scatter(point):
    r = 5
    n = random.randint(3, 5)
    points = []
    for i in range(n):
        radius = random.randint(r, 3 * r)
        angle = random.randint(0, 360)
        x = radius * np.cos(2 * np.pi * angle / 360) + point[0]
        y = radius * np.sin(2 * np.pi * angle / 360) + point[1]
        points.append((x, y))
    return points


def set_and_launch_pygame_display():
    pygame.init()
    r = 3
    screen = pygame.display.set_mode((600, 400), pygame.RESIZABLE)
    screen.fill('white')
    pygame.display.update()
    points = []
    points_color = []
    flag = True
    mouse_button_down = False
    while flag:
        for event in pygame.event.get():
            color = 'black'
            is_point_single = False
            if event.type == pygame.MOUSEMOTION:
                if event.buttons[1]:
                    color = 'red'
                    is_point_single = True
                if event.buttons[2]:
                    color = 'mediumorchid'

            if event.type == pygame.QUIT:
                pygame.quit()
                flag = False
                break
            if event.type == pygame.WINDOWEXPOSED:
                screen.fill('white')
                for point, color in zip(points, points_color):
                    pygame.draw.circle(screen,
                                       color=color, center=point,
                                       radius=r)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 2:
                    color = 'red'
                    is_point_single = True
                if event.button == 3:
                    color = 'mediumorchid'
                if event.button == 1 or 2 or 3:
                    mouse_button_down = True
            if event.type == pygame.MOUSEBUTTONUP:
                mouse_button_down = False
            if mouse_button_down:
                if is_point_single:
                    coord = event.pos

                    if len(points):
                        if dist(coord, points[-1]) > 5 * r:
                            points.append(coord)

                            points_color.append(color)
                            pygame.draw.circle(screen,
                                               color=color, center=coord,
                                               radius=r)
                    else:
                        points.append(coord)
                        pygame.draw.circle(screen,
                                           color=color, center=coord,
                                           radius=r)

                else:
                    coord = event.pos
                    points_new = scatter(coord)
                    colors_new = [color for _ in range(len(points_new))]

                    if len(points):
                        if dist(coord, points[-1]) > 5 * r:
                            points.append(coord)

                            points.extend(points_new)
                            points_color.extend(colors_new)
                            for point in points_new:
                                pygame.draw.circle(screen,
                                                   color=color, center=point,
                                                   radius=r)

                            pygame.draw.circle(screen,
                                               color=color, center=coord,
                                               radius=r)
                    else:
                        points.append(coord)
                        pygame.draw.circle(screen,
                                           color=color, center=coord,
                                           radius=r)
            pygame.display.update()
    return points


def fitSVC(random_state):
    clf = svm.SVC(kernel='linear', C=1000)
    x, y = make_blobs(n_samples=40, centers=2, random_state=random_state, center_box=(-5.0, 5.0))
    # x, y = [x[0] for x in points], [x[1] for x in points]
    clf.fit(x, y)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none')
    plt.show()
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    return clf


def prompt_point():
    input_string = input("Enter two coordinates separated with space: ")
    return list(map(lambda x: float(x), input_string.split()))


def draw_with_user_point(random_state, clf, user_points, user_point_classes):
    x, y = make_blobs(n_samples=40, centers=2, random_state=random_state, center_box=(-5.0, 5.0))
    user_point = np.array(prompt_point()).reshape(1, 2)
    user_points = np.append(user_points, user_point, axis=0)
    user_point_classes = np.append(
        user_point_classes,
        clf.predict(user_point),
        axis=0)
    plt.scatter(
        np.append(x, user_points, axis=0)[:, 0],
        np.append(x, user_points, axis=0)[:, 1],
        c=np.append(y, user_point_classes, axis=0),
        s=30,
        cmap=plt.cm.Paired)
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none')
    plt.show()

    return user_points, user_point_classes


def launch():
    # points = set_and_launch_pygame_display()
    random_state = 9
    clf = fitSVC(random_state)
    user_points = np.empty((0, 2))
    user_point_classes = np.empty((0))
    while True:
        user_points, user_point_classes = draw_with_user_point(random_state, clf, user_points, user_point_classes)


if __name__ == '__main__':
    launch()

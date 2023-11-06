from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from itertools import combinations
import random
import numpy


def draw(points, normalized_points, classes):
    fig, axs = plt.subplots(4, 3)
    comb = list(combinations((0, 1, 2, 3), 2))

    current_combination_index = 0
    for i in range(2):
        for j in range(3):
            current_combination = comb[current_combination_index]
            axs[i][j].scatter(points[:, current_combination[0]], points[:, current_combination[1]], c=classes)
            current_combination_index += 1

    current_combination_index = 0
    for i in range(2, 4):
        for j in range(3):
            current_combination = comb[current_combination_index]
            axs[i][j].scatter(normalized_points[:, current_combination[0]], normalized_points[:, current_combination[1]], c=classes)
            current_combination_index += 1

    plt.show()


def replace_if_any_value_is_bigger(from_values, to_values):
    for i in range(len(from_values)):
        if from_values[i] > to_values[i]:
            to_values[i] = from_values[i]


def replace_if_any_value_is_less(from_values, to_values):
    for i in range(len(from_values)):
        if from_values[i] < to_values[i]:
            to_values[i] = from_values[i]


def find_min_and_max_features_values(data):
    max_values = data[0].copy()
    min_values = data[0].copy()
    for flower in data:
        replace_if_any_value_is_bigger(flower, max_values)
        replace_if_any_value_is_less(flower, min_values)

    return max_values, min_values


def get_normalized_data(data):
    max_values, min_values = find_min_and_max_features_values(data)
    normalized_data = numpy.ndarray(data.shape)
    for point_index in range(data.shape[0]):
        point = data[point_index]
        dimension = len(point)
        for i in range(dimension):
            normalized_data[point_index, i] = (point[i] - min_values[i]) / (max_values[i] - min_values[i])

    return normalized_data


def get_distance(point_A, point_B):
    differences_squares_sum = 0
    for point_A_coordinate, point_B_coordinate in zip(point_A, point_B):
        differences_squares_sum += (point_A_coordinate - point_B_coordinate) ** 2

    return differences_squares_sum ** 0.5


def calculate_accuracy(predicted_classes, target_class):
    for predicted_class, class_probability in predicted_classes.items():
        if predicted_class == target_class:
            return class_probability

    return 0.0


def predict_classes(known_points, k_value, new_point):
    nearest_neighbours = sorted(
        known_points,
        key = lambda point: get_distance(point[0], new_point),
        reverse = True)[0:k_value]

    classes = {}
    for _, neighbour_class in nearest_neighbours:
        classes.setdefault(neighbour_class, 0)
        classes[neighbour_class] += 1
    for _class, neighbours_count in classes.items():
        classes[_class] /= k_value

    return classes


def find_fitting_k_parameter_value(start_from, to, training_dataset, test_dataset):
    max_accuracy = 0.0
    best_k_value = 0

    for k in range(start_from, to):
        accuracy_sum = 0.0
        for point, target_class in test_dataset:
            predicted_classes = predict_classes(training_dataset, k, point)
            accuracy_sum += calculate_accuracy(predicted_classes, target_class)

        current_accuracy = accuracy_sum / len(test_dataset)
        if current_accuracy > max_accuracy:
            max_accuracy = current_accuracy
            best_k_value = k

    return best_k_value


def enter_new_point(training_set, k_value):
    coordinates = input("Enter 4 coordinates from 0.0 to 1.0 each separated by space: ")
    point = list(map(
        lambda coordinate: float(coordinate),
        coordinates.strip().split(" ")))

    predicted_classes = predict_classes(training_set, k_value, point)

    print(f"For point {point}")
    for _class, probability in predicted_classes.items():
        print(f"class {_class} with {probability * 100}%")


def shuffle(data, _from, to):
    result = data[_from:to]
    random.shuffle(result)
    return result


def main():
    irises = load_iris()
    points = irises.get('data')
    normalized_points = get_normalized_data(points)
    classes = irises.get('target')

    draw(points, normalized_points, classes)

    class_size = 50
    data_set = list(zip(normalized_points, classes))
    class1 = shuffle(data_set, class_size * 0, class_size * 1)
    class2 = shuffle(data_set, class_size * 1, class_size * 2)
    class3 = shuffle(data_set, class_size * 2, class_size * 3)

    test_set_border = int(class_size * 0.3 - 1)
    test_set = class1[0:test_set_border] + class2[0:test_set_border] + class3[0:test_set_border]
    random.shuffle(test_set)

    training_set = class1[test_set_border:class_size] + class2[test_set_border:class_size] + class3[test_set_border:class_size]
    random.shuffle(training_set)

    k_value = find_fitting_k_parameter_value(1, class_size * 3 - 1, training_set, test_set)
    print(f"Best K value: {k_value}")

    enter_new_point(training_set, k_value)


if __name__ == '__main__':
    main()

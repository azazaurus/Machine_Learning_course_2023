from GA import GA


def main():
    constants_string = input("Enter Diophantine equation coefficients. For example \"10 34 0 12 6 56\": ")
    diophantine_constants = list(map(lambda x: int(x), constants_string.split()))
    genetic_algorithm = GA(diophantine_constants, random_seed=1)
    genetic_algorithm.create_initial_population()

    iterations_count = 0
    while True:
        genetic_algorithm.make_next_generation()

        answer, fitness_value = genetic_algorithm.get_answer_with_best_fitness()
        if fitness_value == 0:
            break

        iterations_count += 1
        if iterations_count % 1000 == 0:
            print(f"Current best answer {answer} with fitness value {fitness_value}")

    print(f"Answer: {answer}")


if __name__ == '__main__':
    main()

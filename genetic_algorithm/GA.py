import random


class GA:
    diafantin_constants = []
    max_parameter_value: int
    population = []
    population_size: int
    mutation_probability: float

    def __init__(self, diafantin_constants, population_size=500, mutation_probability=0.05, random_seed=None):
        self.diafantin_constants = diafantin_constants
        self.max_parameter_value = get_max_parameter_value(diafantin_constants)
        self.population_size = population_size
        self.mutation_probability = mutation_probability

        if random_seed is not None:
            random.seed(random_seed)

    def get_answer_with_best_fitness(self):
        best_answer = None
        best_fitness_value = None

        for answer in self.population:
            fitness_value = self.fitness(answer)
            if fitness_value == 0:  # 0 is the best possible fitness value
                return answer, fitness_value

            if best_fitness_value is None or fitness_value < best_fitness_value:
                best_answer = answer
                best_fitness_value = fitness_value

        return best_answer, best_fitness_value

    def fitness(self, parameters):
        result_sum = 0
        for i in range(len(parameters)):
            result_sum += parameters[i] * self.diafantin_constants[i]

        return abs(result_sum - self.diafantin_constants[-1])

    def create_initial_population(self):
        for i in range(self.population_size):
            self.population.append(
                [self.generate_random_parameter_value()
                 for _ in range(len(self.diafantin_constants) - 1)])

    def calculate_survival_probability_per_individual(self):
        inverse_fitness_values = []
        for individual in self.population:
            fitness_value = self.fitness(individual)
            inverse_fitness_values.append(1 / fitness_value if fitness_value != 0 else None)

        survival_probability_per_individual = []
        for inverse_fitness_value in inverse_fitness_values:
            if inverse_fitness_value is None:
                survival_probability_per_individual.append(1)
                continue

            survival_probability_per_individual.append(inverse_fitness_value)

        return survival_probability_per_individual

    def selection(self):
        ancestors = []
        survival_probability_per_individual = self.calculate_survival_probability_per_individual()
        for i in range(len(survival_probability_per_individual)):
            if random_bool(survival_probability_per_individual[i]):
                ancestors.append(self.population[i])

        return ancestors

    def crossing(self, ancestors):
        descendants_number = self.population_size - len(ancestors)
        descendants = []
        for i in range(descendants_number):
            first_parent_index = random.randint(0, self.population_size - 1)
            second_parent_index = random.randint(0, self.population_size - 1)
            first_parent = self.population[first_parent_index]
            second_parent = self.population[second_parent_index]

            parameter_count_from_first_parent = random.randint(1, len(first_parent) - 1)
            descendant = first_parent[:parameter_count_from_first_parent] \
                         + second_parent[parameter_count_from_first_parent:]
            self.mutate(descendant)
            descendants.append(descendant)

        return descendants

    def make_next_generation(self):
        ancestors = self.selection()
        descendants = self.crossing(ancestors)
        self.population = ancestors + descendants

    def mutate(self, individual):
        for i in range(len(individual)):
            parameter = individual[i]
            new_parameter = 0
            exponent = 1
            while parameter > 0:
                # here genes are bits of the parameter
                gene = parameter & 1  # extract gene
                if random_bool(self.mutation_probability):
                    gene ^= 1  # mutate gene

                # write gene
                if gene != 0:
                    new_parameter |= exponent
                exponent <<= 1

                parameter >>= 1  # move to next gene

            individual[i] = new_parameter

    def generate_random_parameter_value(self):
        return random.randint(0, self.max_parameter_value)


def random_bool(true_probability):
    return random.random() < true_probability


def get_max_parameter_value(diafantin_constants):
    diafantin_result = diafantin_constants[-1]
    diafantin_non_zero_coefficients = filter(lambda x: x != 0, diafantin_constants[:-1])
    return diafantin_result // min(diafantin_non_zero_coefficients)

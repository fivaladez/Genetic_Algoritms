"""Contains the logic to use and create a genetic algorithm to solve the traveler problem."""
import numpy as np
import math


class GeneticAlgorithm:
    """"""

    def __init__(self, population_size=1, chromosome_size=14):
        """"""
        self.mapping_table = {  # {city: coordinates (x, y)}
            1: (1, 7), 2: (2, 5), 3: (4, 4), 4: (2, 3), 5: (3, 2),
            6: (1, 1), 7: (5, 1), 8: (7, 3), 9: (6, 6), 10: (10, 5),
            11: (9, 8), 12: (13, 6), 13: (12, 3), 14: (13, 1)
        }
        self.POPULATION_SIZE = abs(int(population_size))
        self.CHROMOSOME_SIZE = abs(int(chromosome_size))
        self.best_chromosome = list()
        self.aptitude_function_history = np.array([], dtype=np.uint8)

    def get_random_population(self):
        """"""
        # Getting initial arrays
        random_population = np.empty((0, self.CHROMOSOME_SIZE), dtype=np.uint8)
        random_chromosome = np.arange(1, self.CHROMOSOME_SIZE + 1, dtype=np.uint8)

        # Adding shuffle arrays to random_populations
        for _ in range(0, self.POPULATION_SIZE):
            np.random.shuffle(random_chromosome)
            random_population = np.append(random_population, [random_chromosome], axis=0)

        """
        # Adding a '0' at the end of each array (aptitude function)
        aptitude_initial_values = np.zeros((self.POPULATION_SIZE, 1), dtype=np.uint8)
        random_population = np.append(random_population, aptitude_initial_values, axis=1)
        """

        return random_population

    def get_aptitude_function(self, population):
        """"""
        def get_distance(p1, p2):
            """Get distance from two given points"""
            return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

        aptitude_function = np.empty(0, dtype=np.float32)
        population_index = 0
        while population_index < self.POPULATION_SIZE:
            summation, chromosome_index = 0, 0
            while chromosome_index + 1 < self.CHROMOSOME_SIZE:
                distance = get_distance(
                    self.mapping_table.get(population[population_index][chromosome_index]),
                    self.mapping_table.get(population[population_index][chromosome_index + 1])
                    )
                summation += np.float32(distance)
                chromosome_index += 1
            aptitude_function = np.append(aptitude_function, [summation], axis=0)
            population_index += 1

        return aptitude_function

    def get_next_generation(self, population):
        """"""
        aptitude_function = self.get_aptitude_function(population)
        return None, None

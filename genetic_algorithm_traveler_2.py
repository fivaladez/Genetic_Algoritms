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

    def get_tournament_winner(self, population, aptitude_function):
        """"""
        # Getting contender indexes
        percentage = 0.05
        n_contenders = int(self.POPULATION_SIZE * percentage)
        n_contenders = 1 if n_contenders < 1 else n_contenders
        contenders_indexes = np.random.choice(self.POPULATION_SIZE, n_contenders)

        # Looking for the winner
        winner_index = -1
        for index in contenders_indexes:
            if winner_index != -1:
                if aptitude_function[winner_index] > aptitude_function[index]:
                    winner_index = index
            else:
                winner_index = index

        return population[winner_index]

    def reproduction(self, chromosome):
        """"""
        child_chromosome = np.copy(chromosome)
        option = np.random.randint(0, 2, dtype=np.uint8)  # Randomly select a reproduction option

        if option == 0:  # Reversing a chunk from the array
            while True:
                start_index = np.random.randint(0, self.CHROMOSOME_SIZE, dtype=np.uint8)
                end_index = np.random.randint(start_index, self.CHROMOSOME_SIZE, dtype=np.uint8)
                if start_index < end_index:
                    break

            # Getting inverted section -> [1,2,3] -> [3,2,1]
            revert_index = start_index - 1
            if revert_index > 0:
                inverted_chunk = np.copy(child_chromosome[end_index:(start_index - 1):-1])
            else:
                inverted_chunk = np.copy(child_chromosome[end_index::-1])

            # print("Inverted chunk {} - ({},{})".format(inverted_chunk, start_index, end_index))

            swap_index = 0
            for idx in range(start_index, end_index + 1):
                child_chromosome[idx] = np.copy(inverted_chunk[swap_index])
                swap_index += 1

        else:
            # Getting a random index for chunks A and B
            while True:
                start_index_a = np.random.randint(0, self.CHROMOSOME_SIZE - 3, dtype=np.uint8)
                end_index_a = np.random.randint(start_index_a, self.CHROMOSOME_SIZE - 2,
                                                dtype=np.uint8)
                if start_index_a > end_index_a:
                    continue  # Try again

                chunk_size = end_index_a - start_index_a

                if end_index_a + 1 >= self.CHROMOSOME_SIZE - chunk_size:
                    continue
                start_index_b = np.random.randint(end_index_a + 1,
                                                  self.CHROMOSOME_SIZE - chunk_size, dtype=np.uint8)
                end_index_b = start_index_b + chunk_size
                if end_index_b >= self.CHROMOSOME_SIZE:
                    continue  # Try again

                break  # Everything went ok

            # Getting chunks
            first_chunk = np.copy(child_chromosome[start_index_a: end_index_a + 1])
            second_chunk = np.copy(child_chromosome[start_index_b:end_index_b + 1])

            # print("1° chunk {}, Indexes ({},{})".format(first_chunk, start_index_a, end_index_a))
            # print("2° chunk {}, Indexes ({},{})".format(second_chunk, start_index_b, end_index_b))

            # Swapping the chunks in the child chromosome
            swap_index = 0
            for idx in range(start_index_a, end_index_a + 1):
                child_chromosome[idx] = np.copy(second_chunk[swap_index])
                child_chromosome[start_index_b] = np.copy(first_chunk[swap_index])
                start_index_b += 1
                swap_index += 1

        return child_chromosome

    def get_best_from_population(self, population, aptitude_function):
        """"""
        # Finding the best from the current population
        index = 0
        best_index = index

        while index < self.POPULATION_SIZE:
            if aptitude_function[best_index] > aptitude_function[index]:
                best_index = index
            index += 1

        return population[best_index], aptitude_function[best_index]

    def get_next_generation(self, population):
        """"""
        aptitude_function = self.get_aptitude_function(population)
        childes_population = np.empty((0, self.CHROMOSOME_SIZE), dtype=np.uint8)

        for i in range(self.POPULATION_SIZE):
            parent_chromosome = self.get_tournament_winner(population, aptitude_function)
            child_chromosome = self.reproduction(parent_chromosome)
            childes_population = np.append(childes_population, [child_chromosome], axis=0)
            # print("Parent: {}, Child: {}\n\n".format(parent_chromosome, child_chromosome))

        childes_aptitude_function = self.get_aptitude_function(childes_population)

        # Saving and comparing against the best chromosome from history
        # Example: best_chromosome = [chromosome, aptitude_function]
        best_chromosome = self.get_best_from_population(
            childes_population, childes_aptitude_function)
        if self.best_chromosome:
            if self.best_chromosome[1] > best_chromosome[1]:
                self.best_chromosome = best_chromosome
        else:
            self.best_chromosome = best_chromosome

        # Save the best aptitude function from the current population
        self.aptitude_function_history = np.append(self.aptitude_function_history,
                                                   best_chromosome[1])

        return childes_population, childes_aptitude_function

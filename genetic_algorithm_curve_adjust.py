"""This file contains the logic to use and create a genetic algorithm"""
import matplotlib.pyplot as plt
import numpy as np


class GeneticAlgorithm:
    """"""

    def __init__(self, x, y, population_size=1, chromosome_size=7):
        """"""
        self.X = float(x)
        self.Y = float(y)
        self.POPULATION_SIZE = int(population_size) if population_size > 0 else abs(population_size)
        self.CHROMOSOME_SIZE = int(chromosome_size) if chromosome_size > 0 else abs(chromosome_size)
        self.SHAPE = (self.POPULATION_SIZE, self.CHROMOSOME_SIZE)
        self.best_chromosome = list()  # [chromosome, aptitude_function]
        self.weight = 25.5
        self.aptitude_function_history = np.array([], dtype=np.float32)

    def get_aptitude_function(self, population):
        """"""
        aptitude_function_population = np.array([], dtype=np.float32)
        population_index = 0

        while population_index < self.POPULATION_SIZE:
            chromosome = population[population_index]
            chromosome_index = 0
            expression = 0  # (A'/weight) * x^6 ... (G'/weight) * x^0
            exponent = self.CHROMOSOME_SIZE - 1  # x^6 ... x^0

            while chromosome_index < self.CHROMOSOME_SIZE and exponent >= 0:
                expression += (chromosome[chromosome_index] / self.weight) * pow(self.X, exponent)
                chromosome_index += 1
                exponent -= 1

            aptitude_function = np.float32(self.Y - expression)
            aptitude_function_population = np.append(aptitude_function_population,
                                                     [aptitude_function])
            population_index += 1

        return aptitude_function_population

    def get_equation_y(self, chromosome, arr_x):
        """"""
        constants = chromosome / self.weight  # [A ... G] - where A = A' / weight
        arr_y = np.array([])
        for x in arr_x:
            exponent = self.CHROMOSOME_SIZE - 1  # x^6 ... x^0
            expression = 0
            while exponent >= 0:
                # (A'/weight) * x^6 ... (G'/weight) * x^0
                expression += constants[self.CHROMOSOME_SIZE - exponent - 1] * pow(x, exponent)
                exponent -= 1
            arr_y = np.append(arr_y, [expression])

        return arr_y

    def get_y(self, chromosome, x):
        """"""
        constants = chromosome / self.weight  # [A ... G] - where A = A' / weight
        exponent = self.CHROMOSOME_SIZE - 1  # x^6 ... x^0
        y = 0
        while exponent >= 0:
            # (A'/weight) * x^6 ... (G'/weight) * x^0
            y += constants[self.CHROMOSOME_SIZE - exponent - 1] * pow(x, exponent)
            exponent -= 1

        return y

    def get_random_population(self):
        """"""
        random_population = np.random.randint(low=0, high=255, dtype=np.uint8,
                                              size=(self.POPULATION_SIZE, self.CHROMOSOME_SIZE))
        return random_population

    def get_next_generation(self, population):
        """"""
        aptitude_function = self.get_aptitude_function(population)
        childes_population = np.empty((0, self.CHROMOSOME_SIZE), dtype=np.uint8)
        iterations = self.POPULATION_SIZE // 2

        for i in range(iterations):
            # Getting father and mother chromosomes
            father_chromosome = self.get_tournament_winner(population, aptitude_function)
            mother_chromosome = self.get_tournament_winner(population, aptitude_function)

            # Reproduce parents to get childes
            son, daughter = self.reproduction(father_chromosome, mother_chromosome)
            print("Parents: {}, {}".format(father_chromosome, mother_chromosome))
            print("Childes: {}, {}\n".format(son, daughter))

            # Add childes as rows for the new population (new generation)
            childes_population = np.append(childes_population, [son], axis=0)
            childes_population = np.append(childes_population, [daughter], axis=0)

        if (iterations * 2) < self.POPULATION_SIZE:  # Repeat one more time with only one child
            # Getting father and mother chromosomes
            father_chromosome = self.get_tournament_winner(population, aptitude_function)
            mother_chromosome = self.get_tournament_winner(population, aptitude_function)

            # Reproduce parents to get childes
            son, daughter = self.reproduction(father_chromosome, mother_chromosome)
            print("Parents: {}, {}".format(father_chromosome, mother_chromosome))
            print("Childes: {}, {}\n".format(son, daughter))

            # Add childes as rows for the new population (new generation)
            childes_population = np.append(childes_population, [son], axis=0)

        # Getting the new aptitude function from new population
        aptitude_function = self.get_aptitude_function(childes_population)

        # Saving and comparing against the best chromosome from history
        # Example: best_chromosome = [chromosome, aptitude_function]
        best_chromosome = self.get_best_from_population(childes_population, aptitude_function)
        if self.best_chromosome:
            if abs(self.best_chromosome[1]) > abs(best_chromosome[1]):
                self.best_chromosome = best_chromosome
        else:
            self.best_chromosome = best_chromosome

        # Save the best aptitude function from the current population
        self.aptitude_function_history = np.append(self.aptitude_function_history,
                                                   abs(best_chromosome[1]))

        return childes_population, aptitude_function

    def get_best_from_population(self, population, aptitude_function):
        """"""
        # Finding the best from the current population
        index = 0
        best_index = index

        while index < self.POPULATION_SIZE:
            if abs(aptitude_function[best_index]) > abs(aptitude_function[index]):
                best_index = index
            index += 1

        return population[best_index], aptitude_function[best_index]

    def get_tournament_winner(self, population, aptitude_function):
        """"""
        # Getting contender indexes
        percentage = 0.05
        n_contenders = int(self.POPULATION_SIZE * percentage)
        n_contenders = 1 if n_contenders < 1 else n_contenders
        contenders_indexes = np.random.randint(0, self.POPULATION_SIZE, n_contenders)

        # Looking for the winner
        winner_index = contenders_indexes[0]
        for index in contenders_indexes:
            if abs(aptitude_function[winner_index]) > abs(aptitude_function[index]):
                winner_index = index

        return population[winner_index]

    def reproduction(self, chromosome_x, chromosome_y):
        """"""
        father = np.copy(chromosome_x)
        mother = np.copy(chromosome_y)
        cut_point = np.random.randint(0, 8 * (self.CHROMOSOME_SIZE - 1), dtype=np.uint8)

        if cut_point % 8 == 0:
            # Getting cut index - [0 ... index ... -1]
            index = int((cut_point / 8) - 1)

            # Adding the chunks - [father ... mother]
            son = np.copy(father[:index + 1])
            son = np.append(son, np.copy(mother[index + 1:]))

            # Adding the chunks - [mother ... father]
            daughter = np.copy(mother[:index + 1])
            daughter = np.append(daughter, np.copy(father[index + 1:]))

            print("Index {}, Son {}, Daughter {}".format(index, son, daughter))

        else:
            # Getting indexes - [first_part ... cut_part ... second_part]
            first_index = cut_point // 8
            second_index = first_index + 2  # Skipping the cut point

            # Getting cut points - 1 byte - 8 bits - 0...255
            father_cut_point = father[first_index + 1]
            mother_cut_point = mother[first_index + 1]

            # Getting bits from each byte - Example: 1 byte = [2 bits: 6 bits]
            first_cut_index = np.uint8(cut_point % 8)
            second_cut_index = np.uint8(8 - first_cut_index)

            """ Example:
             father_cut_point = 134, mother_cut_point = 203
             first_cut_index = 2, second_cut_index = 6 -> Bits from a Byte(8 bits)
             Son = (left first 2 bits from 134 => 128) + (right first 6 bits from 203 => 11)
             Daughter = (left first 2 bits from 203 => 192) + (right first 6 bits from 134 => 6)
             Son = (128 + 11) = 139, Daughter = (192 + 6) = 198
            """

            # Getting the new cut part for son chromosome
            first_cut_part = np.uint8((father_cut_point >> second_cut_index) << second_cut_index)
            second_cut_part = np.uint8((mother_cut_point << first_cut_index) >> first_cut_index)
            son_cut_point = first_cut_part + second_cut_part

            # Getting the new cut part for daughter chromosome
            first_cut_part = np.uint8((mother_cut_point >> second_cut_index) << second_cut_index)
            second_cut_part = np.uint8((father_cut_point << first_cut_index) >> first_cut_index)
            daughter_cut_point = first_cut_part + second_cut_part

            # Adding the chunks [father ... cut point ... mother]
            son = np.copy(father[:first_index + 1])
            son = np.append(son, [son_cut_point])
            son = np.append(son, np.copy(mother[second_index:]))

            # Adding the chunks [mother ... cut point ... father]
            daughter = np.copy(mother[:first_index + 1])
            daughter = np.append(daughter, [daughter_cut_point])
            daughter = np.append(daughter, np.copy(father[second_index:]))

            print("Cut index {} - Values {}, {}".format(first_index + 1, father[first_index + 1],
                                                        mother[first_index + 1]))
            print("Bits {}, {}".format(first_cut_index, second_cut_index))
            print("Son {}, Daughter {}".format(son_cut_point, daughter_cut_point))

        return son, daughter

    def graph(self, population, aptitude_function, generation):
        """"""
        # Getting coordinates for plotting
        best_chromosome = self.get_best_from_population(population, aptitude_function)
        x = np.arange(-100, 100, dtype=np.int16)
        y = self.get_equation_y(best_chromosome[0], x)

        # Plotting
        plt.figure(1)
        plt.plot(x, y)
        plt.grid(color="gray", linestyle="--", linewidth=1, alpha=.4)
        plt.title("Best Chromosome {}".format(best_chromosome))
        plt.show(block=False)

        plt.figure(2)
        plt.plot(np.arange(0, generation, dtype=np.uint16), self.aptitude_function_history)
        plt.grid(color="gray", linestyle="--", linewidth=1, alpha=.4)
        plt.title("Closest to 0: {}".format(self.aptitude_function_history[-1]))
        plt.show(block=False)
        plt.pause(1)

        plt.figure(1)
        plt.clf()

        plt.figure(2)
        plt.clf()
"""This file contains the logic to use and create a genetic algorithm"""
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
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
        self.best_chromosome = dict()
        self.weight = 25.5

    def get_aptitude_function(self, population):
        """"""
        aptitude_function_population = np.array([])
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

            aptitude_function = self.Y - expression
            aptitude_function_population = np.append(aptitude_function_population,
                                                     [aptitude_function])
            population_index += 1

        return aptitude_function_population

    def get_random_population(self):
        """"""
        random_population = np.random.randint(low=0, high=255, dtype=np.uint8,
                                              size=(self.POPULATION_SIZE, self.CHROMOSOME_SIZE))
        return random_population

    def get_next_generation(self, population):
        """"""
        aptitude_function = self.get_aptitude_function(population)

        for i in range(self.POPULATION_SIZE):
            father_chromosome = self.get_tournament_winner(population, aptitude_function)
            mother_chromosome = self.get_tournament_winner(population, aptitude_function)
            son, daughter = self.reproduction(father_chromosome, mother_chromosome)
            print("Parents: {}, {}".format(father_chromosome, mother_chromosome))
            print("Childes: {}, {}\n".format(son, daughter))

        return None, None

    def get_tournament_winner(self, population, aptitude_function):
        """"""
        # Getting contender indexes
        percentage = 0.05
        n_contenders = int(self.POPULATION_SIZE * percentage)
        n_contenders = 1 if n_contenders < 1 else n_contenders
        contenders_indexes = np.random.randint(0, self.POPULATION_SIZE, n_contenders)

        # Looking for the winner
        winner_index = -1
        for index in contenders_indexes:
            if winner_index != -1:
                winner_difference = abs(0 - aptitude_function[winner_index])
                current_difference = abs(0 - aptitude_function[index])
                if winner_difference > current_difference:
                    winner_index = index
            else:
                winner_index = index

        return population[winner_index]

    def reproduction(self, chromosome_x, chromosome_y):
        """"""
        father = chromosome_x[:]
        mother = chromosome_y[:]
        cut_point = np.random.randint(0, 8 * (self.CHROMOSOME_SIZE - 1), dtype=np.uint8)

        if cut_point % 8 == 0:
            index = int((cut_point / 8) - 1)
            son = father[:index + 1]
            son = np.append(son, mother[index + 1:])
            daughter = mother[:index + 1]
            daughter = np.append(daughter, father[index + 1:])
            return son, daughter
        else:
            # Getting indexes - [first_part ... cut_part ... second_part]
            first_index = np.uint8(cut_point // 8)
            second_index = first_index + 2  # Skipping the cut point

            # Getting cut points - 1 byte - 8 bits - 0...255
            father_cut_point = father[first_index + 1]
            mother_cut_point = mother[first_index + 1]

            # Getting bits from each byte - Example: 1 byte = [2 bits: 6 bits]
            first_cut_index = np.uint8(cut_point % 8)
            second_cut_index = 8 - first_cut_index

            print("Cut index {} - Values {}, {}".format(first_index + 1, father[first_index + 1], mother[first_index + 1]))
            print("Bits {}, {}".format(first_cut_index, second_cut_index))

            # Getting the new cut part for son chromosome
            first_cut_part = np.uint8((father_cut_point >> second_cut_index) << second_cut_index)
            second_cut_part = np.uint8((mother_cut_point << first_cut_index) >> first_cut_index)
            son_cut_point = first_cut_part + second_cut_part

            # Getting the new cut part for daughter chromosome
            first_cut_part = np.uint8((mother_cut_point >> second_cut_index) << second_cut_index)
            second_cut_part = np.uint8((father_cut_point << first_cut_index) >> first_cut_index)
            daughter_cut_point = first_cut_part + second_cut_part

            print(son_cut_point, daughter_cut_point)

            return None, None

    def graph(self, population, aptitude_function):
        """"""
        pass

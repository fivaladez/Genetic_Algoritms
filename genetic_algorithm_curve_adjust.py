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
        self.best_chromosome = -1
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
            pass
            winner_chromosome = self.get_tournament_winner(population, aptitude_function)

        return None, None

    def get_tournament_winner(self, population, aptitude_function):
        """"""
        # Getting contender indexes
        percentage = 0.05
        n_contenders = int(self.POPULATION_SIZE * percentage)
        n_contenders = 1 if n_contenders < 1 else n_contenders
        contenders_indexes = np.random.randint(0, self.POPULATION_SIZE, n_contenders)

        # Looking for the winner


        return None

    def graph(self, population, aptitude_function):
        """"""
        pass


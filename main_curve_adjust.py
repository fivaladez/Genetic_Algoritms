import numpy as np
from genetic_algorithm_curve_adjust import GeneticAlgorithm

if __name__ == '__main__':
    # Fixed values
    X, Y = 1.5, 174.57
    POPULATION_SIZE = 100
    CHROMOSOME_SIZE = 7
    GENERATIONS = 1

    # Getting random initial population
    curve_adjust = GeneticAlgorithm(X, Y, POPULATION_SIZE, CHROMOSOME_SIZE)
    parent_population = curve_adjust.get_random_population()

    # Getting generations
    for generation in range(1, GENERATIONS + 1):
        child_population, aptitude_function = curve_adjust.get_next_generation(parent_population)
        # curve_adjust.graph(child_population, aptitude_function)
        # parent_population = child_population

    # Presenting best chromosome
    input("Pause")

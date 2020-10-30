from genetic_algorithm_traveler_2 import GeneticAlgorithm
import numpy as np

if __name__ == '__main__':
    POPULATION_SIZE = 200
    GENERATIONS = 100

    # Getting initial random population
    traveler = GeneticAlgorithm(POPULATION_SIZE)
    parent_population = traveler.get_random_population()
    print(parent_population.shape)

    for generation in range(1, GENERATIONS + 1):
        child_population, aptitude_function = traveler.get_next_generation(parent_population)
        # traveler.graph(child_population, aptitude_function, generation)
        parent_population = child_population
        # print("\n\t==================================================\n\n")

    print("Best from History {}".format(traveler.best_chromosome))
    print("Aptitude function History {}".format(traveler.aptitude_function_history))
    input("Pause")
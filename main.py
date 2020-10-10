from genetic_algorithm import GeneticAlgorithm as GA

if __name__ == '__main__':
    # Getting initial random population
    genetic_algorithm = GA(100, 14)
    parent_population = genetic_algorithm.get_random_population()
    print("\tInitial population sizes: [{}, {}]".format(len(parent_population),
                                                        len(parent_population[0])))

    # Adding at the end of each chromosome the summation of the distances between points
    genetic_algorithm.add_aptitude_function(parent_population)
    print("\tNew population sizes: [{}, {}]".format(len(parent_population),
                                                    len(parent_population[0])))

    # Get the next generation
    population_child = genetic_algorithm.get_next_generation(population=parent_population,
                                                             childes=100)
    print("\nChild population sizes: [{}, {}]\n".format(len(population_child),
                                                        len(population_child[0])))

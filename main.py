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
    n_generations = 100
    for i in range(0, n_generations):
        child_population = genetic_algorithm.get_next_generation(population=parent_population,
                                                                 childes=100)
        print("Genaration #{}".format(i))
        print("Child population sizes: [{}, {}]\n".format(len(child_population),
                                                          len(child_population[0])))
        genetic_algorithm.add_aptitude_function(child_population)
        genetic_algorithm.graph(child_population, i)
        parent_population = child_population

    input("Pause")

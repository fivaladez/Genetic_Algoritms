import genetic_algorithm as ga

if __name__ == '__main__':
    # Get initial population
    population = ga.Population(100, 14).get_population()
    """ == Debug Code ==
    for chromosome in population:
        print(chromosome)
    """
    print("\n[{}, {}]".format(len(population), len(population[0])))

    population_parent = population
    population_child = ga.Generation().get_next_generation(population=population_parent,
                                                           iterations=1)
    """ == Debug code ==
    for chromosome_child in population_child:
        print(chromosome_child)
    """
    print("\n[{}, {}]".format(len(population_child), len(population_child[0])))

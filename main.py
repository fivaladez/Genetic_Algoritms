import genetic_algorithm as ga


if __name__ == '__main__':
    # Get initial population
    population = ga.Population(1, 5).get_population()
    # """ == Debug Code ==
    for chromosome in population:
        print(chromosome)
    # """
    print("\n[{}, {}]".format(len(population), len(population[0])))

    population_parent = population



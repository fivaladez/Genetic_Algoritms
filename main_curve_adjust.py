from genetic_algorithm_curve_adjust import GeneticAlgorithm

if __name__ == '__main__':
    # Fixed values 77.3781, 76.96017
    X, Y = 1.5, 77.3781
    POPULATION_SIZE = 100
    CHROMOSOME_SIZE = 7
    GENERATIONS = 20
    """
    # A=1,    B=2,    C=6,     D=3.2,  E=0.5,  F=1,    G=7
    # A'=25,  B'=51,  C'=153,  D'=81,  E'=12,  F'=25,  G'=178
    # Where A' = A * 25.5
    """

    # Getting random initial population
    curve_adjust = GeneticAlgorithm(X, Y, POPULATION_SIZE, CHROMOSOME_SIZE)
    parent_population = curve_adjust.get_random_population()

    # Getting generations
    for generation in range(1, GENERATIONS + 1):
        child_population, aptitude_function = curve_adjust.get_next_generation(parent_population)
        curve_adjust.graph(child_population, aptitude_function, generation)
        parent_population = child_population
        print("\n\t==================================================\n\n")

    print("Best from History {}".format(curve_adjust.best_chromosome))
    print("Aptitude function History {}".format(curve_adjust.aptitude_function_history))
    input("Pause")

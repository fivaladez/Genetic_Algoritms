from genetic_algorithm_curve_adjust_2 import GeneticAlgorithm

if __name__ == '__main__':
    CURVE_CONSTANTS = [8, 25, 4, 45, 10, 17, 35]  # A, B, C, D, E, F, G
    WEIGHT = 5
    POPULATION_SIZE = 1000
    GENERATIONS = 5
    MUTATION = 0.03  # Percentage - 10% = 0.1
    ELITISM = True
    """
    # A=8,    B=25,    C=4,     D=45,    E=10,   F=17,    G=35
    # A'=40,  B'=125,  C'=20,   D'=225,  E'=50,  F'=85,   G'=175
    # Where A' = A * weight; weight = 5
    """

    # Getting random initial population
    curve_adjust = GeneticAlgorithm(CURVE_CONSTANTS, WEIGHT, POPULATION_SIZE, MUTATION, ELITISM)
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
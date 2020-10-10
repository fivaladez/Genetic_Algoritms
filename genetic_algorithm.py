"""This file contains the logic to use and create a genetic algorithm"""
import random
import math


class Population:
    """"""

    def __init__(self, chromosomes=1, genes=1):
        """
        Constructor for the Population class.
        :param chromosomes: Integer with the quantity of chromosomes for the population
        :param genes: Integer with the number of elements for each chromosome
        """
        self.n_chromosomes = chromosomes
        self.n_genes = genes

    def aptitude_function(self, chromosome):
        """

        :param chromosome: List with all the genes for the chromosome
        :return: Integer with the summation of the distances between points (Project Specific)
        """
        mapping_table = {
            1: (1, 7), 2: (2, 5), 3: (4, 4), 4: (2, 3), 5: (3, 2),
            6: (1, 1), 7: (5, 1), 8: (7, 3), 9: (6, 6), 10: (10, 5),
            11: (9, 8), 12: (13, 6), 13: (12, 3), 14: (13, 1)
        }

        def get_distance(p1, p2):
            """Get distance from two given points"""
            return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

        result = 0
        gene = 0
        while gene + 1 < self.n_genes:
            distance = get_distance(mapping_table.get(chromosome[gene]),
                                    mapping_table.get(chromosome[gene + 1]))
            result += int(distance)
            """ == Debug Code ==
            print("Chromosome {} - Distance from {} to {} is {}".format(chromosome,
                                                                        chromosome[gene],
                                                                        chromosome[gene+1],
                                                                        distance))
            """
            gene += 1

        return result

    def get_population(self):
        """
        Returns a bi-dimensional array with all the chromosomes from the population
        :return: List of lists with all the chromosomes
        """
        chromosome, gene = 0, 0
        population = list()
        while chromosome < self.n_chromosomes:

            # Fulfill the chromosome
            population.append(list())
            while gene < self.n_genes:
                value = random.randint(1, self.n_genes)
                if value not in population[chromosome]:
                    population[chromosome].append(value)
                    gene += 1

            # Adding the "aptitude function" at the end of the chromosome
            current_chromosome = population[chromosome]
            population[chromosome].append(self.aptitude_function(current_chromosome))

            gene = 0
            chromosome += 1

        return population


class Generation:
    """"""

    def __init__(self, reproduction_operator=1):
        """"""
        self.operator = reproduction_operator

    def get_tournament_winner(self, population, percentage=.05):
        """

        :param population: List with all the chromosomes
        :param percentage: Floating number from 0-1
        :return: List with the selected winner from the tournament (chromosome with less distance)
        """
        n_contenders = int((len(population)) * percentage)
        n_chromosomes = len(population) - 1
        contenders = dict()  # {key=index: value=distance}
        i = 0
        # Getting dictionary of "contenders"
        while i < n_contenders:
            contender_index = random.randint(0, n_chromosomes)
            if contender_index not in contenders:
                # For this step it is assumed that the distance is at the end of the list
                contenders[contender_index] = population[contender_index][-1]
                i += 1

        # Looking for the winner
        winner = -1
        for key, value in contenders.items():
            if winner != -1:
                if winner[-1] > value:
                    winner = population[key]
            else:
                winner = population[key]

        return winner

    def reproduction(self, chromosome):
        """"""
        start = 1
        end = len(chromosome) - 2
        length = random.randint(start, int(end/2))
        first_part = chromosome[start:start+length]
        second_part = chromosome[start+length: start+length+length+1]
        print("Start {} - End {} - Length {} - First part {} - Second part {}".format(
            start, end, length, first_part, second_part
        ))


        return list()

    def get_next_generation(self, population, iterations=1):
        """

        :param population: List with all the chromosomes from the population
        :param iterations: Integer with the number of required chromosome childes
        :return: List with all the new population (childes)
        """
        child_population = list()
        for _ in range(0, iterations):
            winner = self.get_tournament_winner(population)
            print(winner)
            child_chromosome2 = self.reproduction(winner)
            print(child_chromosome2)
            child_population.append(winner)
        return child_population

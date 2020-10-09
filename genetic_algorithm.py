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
            1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (1, 4), 5: (1, 5),
            6: (2, 1), 7: (2, 2), 8: (2, 3), 9: (2, 4), 10: (2, 5),
            11: (3, 1), 12: (3, 2), 13: (3, 3), 14: (3, 4)
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

    def _get_tournament_winner(self, population):
        """"""
        return 1

    def get_next_generation(self, population):
        """"""
        iterations = len(population)
        child_population = list()
        for _ in iterations:
            child_chromosome = self._get_tournament_winner(population)
            child_population.append(child_chromosome)
        return child_population

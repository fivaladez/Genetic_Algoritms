"""This file contains the logic to use and create a genetic algorithm"""
import random
import math


class GeneticAlgorithm:
    """"""

    def __init__(self, chromosomes=1, genes=1):
        """

        :param chromosomes: Integer with the quantity of chromosomes for the population
        :param genes: Integer with the number of elements for each chromosome
        """
        self.n_chromosomes = chromosomes
        self.n_genes = genes

    def add_aptitude_function(self, population):
        """
        Adding at the end of each chromosome the summation of the distances between points
        :param population: List with all the chromosomes for the population
        """

        def get_distance(p1, p2):
            """Get distance from two given points"""
            return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

        mapping_table = {
            1: (1, 7), 2: (2, 5), 3: (4, 4), 4: (2, 3), 5: (3, 2),
            6: (1, 1), 7: (5, 1), 8: (7, 3), 9: (6, 6), 10: (10, 5),
            11: (9, 8), 12: (13, 6), 13: (12, 3), 14: (13, 1)
        }

        chromosome = 0
        while chromosome < self.n_chromosomes:
            summation, gene = 0, 0
            while gene + 1 < self.n_genes:
                distance = get_distance(mapping_table.get(population[chromosome][gene]),
                                        mapping_table.get(population[chromosome][gene + 1]))
                summation += int(distance)
                gene += 1

            population[chromosome].append(summation)
            chromosome += 1

    def get_random_population(self):
        """
        Returns a bi-dimensional array with all the chromosomes from the population
        :return: List of lists with all the chromosomes
        """
        chromosome, gene = 0, 0
        population = list()
        while chromosome < self.n_chromosomes:

            # Fulfill a chromosome
            population.append(list())
            while gene < self.n_genes:
                value = random.randint(1, self.n_genes)
                if value not in population[chromosome]:
                    population[chromosome].append(value)
                    gene += 1

            gene = 0
            chromosome += 1

        return population

    def _get_tournament_winner(self, population, percentage=.05):
        """
        Select random contenders according to the percentage and returns the best one
        :param population: List with all the chromosomes
        :param percentage: Floating number from 0-1
        :return: List with the selected winner from the tournament (chromosome with less distance)
        """
        # Getting dictionary "contenders" -> {key=index: value=distance}
        n_contenders = int(self.n_chromosomes * percentage)
        contenders = dict()
        i = 0
        while i < n_contenders:
            contender_index = random.randint(0, self.n_chromosomes - 1)
            if contender_index not in contenders:
                # For this step it is assumed that the distance is at the end of the list
                contenders[contender_index] = population[contender_index][-1]
                i += 1

        # Looking for the winner
        winner = -1
        for index, distance in contenders.items():
            if winner != -1:
                if winner[-1] > distance:
                    winner = population[index]
            else:
                winner = population[index]

        return winner

    def _reproduction(self, chromosome):
        """"""
        child_chromosome = chromosome[:]
        child_chromosome.pop()  # Removing aptitude_function
        option = random.randrange(0, 2)  # Randomly select a reproduction option
        if option == 0:
            # Getting random points for first chunk excluding first and last item (0, -1)
            while True:
                init_index = random.randint(1, (self.n_genes - 2) // 2)
                end_index = random.randint(1, (self.n_genes - 2) // 2)
                if init_index < end_index:
                    break

            # Getting inverted section -> [1,2,3] -> [3,2,1]
            inverted_chunk = child_chromosome[end_index:(init_index - 1):-1]

            print("Inverted chunk {} - ({},{})".format(inverted_chunk, init_index, end_index))

            swap_index = 0
            for idx in range(init_index, end_index + 1):
                child_chromosome[idx] = inverted_chunk[swap_index]
                swap_index += 1
        else:
            # Getting random points for first chunk excluding first and last item (0, -1)
            while True:
                init_index_a = random.randint(1, (self.n_genes - 2) // 2)
                end_index_a = random.randint(1, (self.n_genes - 2) // 2)
                if init_index_a < end_index_a:
                    break

            # Getting chunk size
            chunk_size = end_index_a - init_index_a

            # Getting a random index for the second chunk
            while True:
                init_index_b = random.randint(end_index_a + 1, self.n_genes - chunk_size - 2)
                end_index_b = init_index_b + chunk_size
                if end_index_b <= (self.n_genes - 2):
                    break

            # Getting chunks
            first_chunk = child_chromosome[init_index_a: end_index_a + 1]
            second_chunk = child_chromosome[init_index_b:end_index_b + 1]

            print("1° chunk {}, Indexes ({},{})".format(first_chunk, init_index_a, end_index_a))
            print("2° chunk {}, Indexes ({},{})".format(second_chunk, init_index_b, end_index_b))

            # Swapping the chunks in the child chromosome
            swap_index = 0
            for idx in range(init_index_a, end_index_a):
                child_chromosome[idx] = second_chunk[swap_index]
                child_chromosome[init_index_b] = first_chunk[swap_index]
                init_index_b += 1
                swap_index += 1

        return child_chromosome

    def get_next_generation(self, population, childes=1):
        """

        :param population: List with all the chromosomes from the population
        :param childes: Integer with the number of required chromosome childes
        :return: List with all the new population (childes)
        """
        child_population = list()
        for child in range(0, childes):
            print("\nChild: {}".format(child))
            winner = self._get_tournament_winner(population)
            print("Winner: {}".format(winner))
            child_chromosome = self._reproduction(winner)
            print("Child:  {}".format(child_chromosome), end="\n\n")
            child_population.append(child_chromosome)
        return child_population

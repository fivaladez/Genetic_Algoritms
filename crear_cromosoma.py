import random


def generate(rows=1, columns=1):
    i, j = 0, 0
    chromosome = list()
    while i < rows:
        chromosome.append([])
        while j < columns:
            n = random.randint(1, columns)
            if n not in chromosome[i]:
                chromosome[i].append(n)
                j += 1
        j = 0
        i += 1
    return chromosome

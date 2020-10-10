import random
import math
import matplotlib.pyplot as plt
import time

# City Coordinates, Fix position for al the cities

citiesCoordinates = {1: (3, 4), 2: (5, 6), 3: (9, 7), 4: (6, 2), 5: (9, 1), 6: (2, 7), 7: (3, 1),
                     8: (7, 5), 9: (1, 5), 10: (7, 3), 11: (9, 5), 12: (5, 9), 13: (4, 2),
                     14: (9, 3), 15: (2, 3), 16: (7, 8), 17: (3, 8), 18: (1, 10), 19: (9, 9),
                     20: (5, 4)}


def randomPopulationCreator():
    "Create population"

    # Generate population with 100 elements

    initPupulation = []

    while len(initPupulation) < 100:
        # Generate new random chromosome
        randChromosome = []
        while len(randChromosome) < 20:
            citie = random.randrange(1, 21, 1)
            if citie not in randChromosome:
                randChromosome.append(citie)
        if randChromosome not in initPupulation:
            initPupulation.append(randChromosome)

    return initPupulation


def selectPlayers(numberOfPlayers, father, previousPlayers):
    "Extract some players out of the whole population"
    newPlayers = []
    while len(newPlayers) < numberOfPlayers:
        option = random.randrange(0, 100, 1)
        if (father[option] not in newPlayers):
            newPlayers.append(father[option])
    if (previousPlayers == newPlayers):
        newPlayers = selectPlayers(numberOfPlayers, father, previousPlayers)
        print("ERROR")

    return newPlayers


def startContest(currentPlayers):
    "Start the contest and see who is more capable for reproduction"
    winnerArray = []
    for playerOption in range(5):
        winnerArray.append(currentPlayers[playerOption][20])
    # Obtain the winner -minimum distance-
    minDistance = min(winnerArray)
    for player in range(5):
        if minDistance in currentPlayers[player]:
            return currentPlayers[player]
    print("ERROR - SOMETHING HAPPENED AT START CONTEST FN")


def calculateDistance(pointA, pointB):
    "Calculates the distance between two points"
    return math.sqrt(math.pow((pointB[1] - pointA[1]), 2) + math.pow((pointB[0] - pointA[0]), 2))


def mutateChromosome(chromosome):
    "Mutates the winner chromosome depending on the type"
    localChromosome = chromosome.copy()
    del localChromosome[20]
    option = random.randrange(0, 2, 1)
    if option == 0:
        # Reproduction Method 1
        sectionForI = 0
        while True:
            firstIndex = random.randint(1, len(localChromosome) - 1)
            endIndex = random.randint(1, len(localChromosome) - 1)
            if firstIndex != endIndex:
                break
        if firstIndex > endIndex:
            firstIndex, endIndex = endIndex, firstIndex
        invertedSection = localChromosome[endIndex:(firstIndex - 1):-1]
        for idx in range(firstIndex, endIndex + 1):
            localChromosome[idx] = invertedSection[sectionForI]
            sectionForI += 1

    else:
        # Getting first indexes
        while True:
            initIndexA = random.randint(0, ((len(localChromosome) - 1) // 2))
            endIndexA = random.randint(0, ((len(localChromosome) - 1) // 2))
            if initIndexA != endIndexA:
                break
        if initIndexA > endIndexA:
            initIndexA, endIndexA = endIndexA, initIndexA


        firstChunk = localChromosome[initIndexA:endIndexA + 1]
        chunkSize = endIndexA - initIndexA

        initIndexB = random.randint(endIndexA + 1, len(localChromosome) - chunkSize - 1)
        secondChunk = localChromosome[initIndexB:initIndexB + chunkSize + 1]

        sectionForI = 0
        for idx in range(initIndexA, endIndexA + 1):
            localChromosome[idx] = secondChunk[sectionForI]
            localChromosome[initIndexB] = firstChunk[sectionForI]
            initIndexB += 1
            sectionForI += 1

    return localChromosome


def graphPath(bestChromosome, numOfGeneration):
    # Graph the best chromosome or individual from the population
    xCoordinates = []
    yCoordinates = []
    for city in range(19):
        xCoordinates.append(citiesCoordinates[bestChromosome[city]][0])
        yCoordinates.append(citiesCoordinates[bestChromosome[city]][1])

    plt.scatter(xCoordinates, yCoordinates, color="blue")
    plt.plot(xCoordinates, yCoordinates, color="blue")
    plt.grid(color="gray", linestyle="--", linewidth=1, alpha=.4)
    plt.title("Trip: " + str(numOfGeneration))
    plt.xlabel("Best of the generation = " + str(bestChromosome[20]))
    plt.show(block=False)
    plt.pause(1)
    plt.clf()


fatherPopulation = randomPopulationCreator()

minToGraph = []
generations = 0

while generations < 100:

    # Calculate distances
    allDistances = {}

    for playerOption in range(100):

        distance = 0;

        for city in range(19):
            distance = distance + calculateDistance(
                citiesCoordinates[fatherPopulation[playerOption][city]],
                citiesCoordinates[fatherPopulation[playerOption][city + 1]])

        fatherPopulation[playerOption].append(distance)
        allDistances[playerOption] = distance

    graphPath(fatherPopulation[(min(allDistances, key=lambda k: allDistances[k]))], generations)
    minToGraph.append(fatherPopulation[(min(allDistances, key=lambda k: allDistances[k]))][20])

    sonPopulation = []

    Players = []

    while len(sonPopulation) < 100:
        Players = selectPlayers(5, fatherPopulation, Players)

        winner = startContest(Players)

        mutatedWinner = mutateChromosome(winner)

        sonPopulation.append(mutatedWinner)

    fatherPopulation = sonPopulation.copy()

    generations += 1

generations = 100
# Graph distance over the total generation
gen = list(range(generations))
plt.grid(color="gray", linestyle="--", linewidth=1, alpha=.4)
plt.plot(gen, minToGraph, color="green")
plt.title("Minimum Distance Over Generations, Minimum distance found:" + str(min(minToGraph)))
plt.ylabel("Total Traveler Distance")
plt.xlabel("Generations")
plt.show()

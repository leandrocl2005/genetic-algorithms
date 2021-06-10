from random import choice, choices, randint, random, randrange
from typing import Callable, List, Tuple
from functools import partial
from matplotlib.text import OffsetFrom

import numpy as np
from numpy.linalg import norm


def f(x, y):
    return 3 * x + 4 * y + random() * 0.1


x_values = [5 * random() for _ in range(100)]
y_values = [5 * random() for _ in range(100)]
f_values = [f(t[0], t[1]) for t in zip(x_values, y_values)]
"""
import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.scatter(x_values, f_values)
plt.subplot(1, 2, 2)
plt.scatter(y_values, f_values)
plt.tight_layout()
plt.show()
"""

# We want find W1 W2 for pred_f(x,y) = W1x + W2y

Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]


def generate_genome(length: int) -> Genome:
    return [randint(-10, 10) for _ in range(length)]


def generate_population(size: int) -> Population:
    return [generate_genome(2) for _ in range(size)]


def fitness(genome: Genome) -> int:
    if len(genome) != 2:
        raise ValueError("genome must be lenght 2")

    if genome[0] < -10 or genome[0] > 10:
        return 0

    if genome[1] < -10 or genome[1] > 10:
        return 0

    f_pred = [
        genome[0] * t[0] + genome[1] * t[1] for t in zip(x_values, y_values)
    ]

    return 1 / (1 + norm(np.array(f_pred) - np.array(f_values)))


def selection_pair(population: Population,
                   fitness_func: FitnessFunc) -> Population:
    return choices(population=population,
                   weights=[fitness_func(genome) for genome in population],
                   k=2)


def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:

    if len(a) != len(b):
        raise ValueError("Genome a and b must be of same lenght")

    return [a[0], b[1]], [b[0], a[1]]


def mutation(genome: Genome, num: int = 1):
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] + choice([0, 0, -1, 1])
    return genome


def run_evolution(populate_func: PopulateFunc,
                  fitness_func: FitnessFunc,
                  fitness_limit: int,
                  selection_func: SelectionFunc = selection_pair,
                  crossover_func: CrossoverFunc = single_point_crossover,
                  mutation_func: MutationFunc = mutation,
                  generation_limit: int = 100) -> Tuple[Population, int]:
    population = populate_func()
    alphas = []

    for i in range(generation_limit):
        population = sorted(population,
                            key=lambda genome: fitness_func(genome),
                            reverse=True)

        if fitness_func(population[0]) >= fitness_limit:
            break

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation

        alphas.append(fitness_func(population[0]))

    return population, i, alphas


population, generations, alphas = run_evolution(
    populate_func=partial(
        generate_population,
        size=10,
    ),
    fitness_func=partial(fitness),
    fitness_limit=0.95,
    generation_limit=100,
)

print(f'number of generations: {generations}')
print(f'best solution: W1 = {population[0][0]}, W2 = {population[0][1]}')
"""
import matplotlib.pyplot as plt

plt.plot(list(range(len(alphas))), alphas)
plt.show()
"""
# population: many backpack solutions

# A population in a given point is called  generation
# fitness function: specie -> specie score
# if specie don't complain the constrains, fitness function returns zero
# make selection to get next generation
# make mutation to get next generation
# make crossover to get next generation

from random import choices, randint, randrange, random
from functools import partial
from typing import Callable, List, Tuple, NamedTuple

# each species in population have a genome: 1001100111
Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]


class Thing(NamedTuple):
    name: str
    value: float
    weight: float


things = [
    Thing('Laptop', 500, 2200),
    Thing('Headphones', 150, 160),
    Thing('Coffee Mug', 60, 360),
    Thing('Notepad', 40, 333),
    Thing('Water Bottle', 30, 192)
]


def generate_genome(length: int) -> Genome:
    return choices([0, 1], k=length)


def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]


def fitness(genome: Genome, things: List[Thing], weight_limit: int) -> int:
    if len(genome) != len(things):
        raise ValueError("genome and things must be of the same lenght")

    weight = 0
    value = 0

    for i, thing in enumerate(things):
        if genome[i] == 1:
            weight += thing.weight
            value += thing.value

        if weight > weight_limit:
            return 0

    return value


def selection_pair(population: Population,
                   fitness_func: FitnessFunc) -> Population:
    return choices(population=population,
                   weights=[fitness_func(genome) for genome in population],
                   k=2)


def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:

    if len(a) != len(b):
        raise ValueError("Genome a and b must be of same lenght")

    lenght = len(a)

    p = randint(1, lenght - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]


def mutation(genome: Genome, num: int = 1, probability: float = 0.5):
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(
            1 - genome[index])
    return genome


def run_evolution(populate_func: PopulateFunc,
                  fitness_func: FitnessFunc,
                  fitness_limit: int,
                  selection_func: SelectionFunc = selection_pair,
                  crossover_func: CrossoverFunc = single_point_crossover,
                  mutation_func: MutationFunc = mutation,
                  generation_limit: int = 100) -> Tuple[Population, int]:
    population = populate_func()

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

    return population, i


def genome_to_things(genome: Genome, things: List[Thing]) -> List[str]:
    if len(genome) != len(things):
        raise ValueError("Genome and things must be same lenght")

    things_in_backpack = []

    for i in range(len(genome)):
        if genome[i] == 1:
            things_in_backpack.append(things[i])

    return things_in_backpack


population, generations = run_evolution(
    populate_func=partial(
        generate_population,
        genome_length=len(things),
        size=10,
    ),
    fitness_func=partial(fitness, things=things, weight_limit=3000),
    fitness_limit=740,
    generation_limit=100,
)

print(f'number of generations: {generations}')
print(f'best solution: {genome_to_things(population[0], things)}')

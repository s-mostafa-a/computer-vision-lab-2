import numpy as np
from code.canny import canny, quality_assessment
from PIL import Image

CANNY_PARAMETERS_NUMBER = 3
SOLUTION_PER_POPULATION = 8
PARENTS_MATING_NUMBER = 4


def cal_pop_fitness(image, weak, strong, ground_truth, population):
    fitness = []
    for gene_i in range(population.shape[0]):
        detection = canny(image=image, weak=weak, strong=strong,
                          cutoff_frequency=population[gene_i, 0],
                          alpha=population[gene_i, 1], low=population[gene_i, 2])
        true_positive_rate, _, accuracy = quality_assessment(detection=detection,
                                                             ground_truth=ground_truth)
        fitness.append(true_positive_rate + accuracy)
    return np.array(fitness)


def select_mating_pool(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents


def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1] / 2)
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutation(offspring_crossover):
    for idx in range(offspring_crossover.shape[0]):
        random_value = np.random.randint(-5, 6)
        rnt = np.random.randint(0, CANNY_PARAMETERS_NUMBER)
        if rnt == 0:
            random_value = (random_value // 2) * 2 + 1
            if offspring_crossover[idx, rnt] + random_value > 0:
                offspring_crossover[idx, rnt] = offspring_crossover[idx, rnt] + random_value
        elif rnt == 1:
            if 10 < offspring_crossover[idx, rnt] + random_value < 80:
                offspring_crossover[idx, rnt] = offspring_crossover[idx, rnt] + random_value
        elif rnt == 2:
            offspring_crossover[idx, rnt] = offspring_crossover[idx, rnt] + random_value
    return offspring_crossover


if __name__ == '__main__':
    img = np.array(Image.open("../data/source/100075-original.jpg").convert('L'))
    gnd_trt = np.array(Image.open("../data/source/100075-reference.jpg").convert('L'))
    stn = 255
    wk = 128
    gnd_trt = np.invert(gnd_trt)
    gnd_trt[gnd_trt > wk] = stn
    gnd_trt[gnd_trt <= wk] = 0
    gnd_trt = gnd_trt / stn

    population_size = (SOLUTION_PER_POPULATION, CANNY_PARAMETERS_NUMBER)
    new_population = np.array([[11, 53, 20],
                               [9, 60, 30],
                               [7, 45, 15],
                               [1, 30, 10],
                               [3, 40, 25],
                               [5, 70, 20],
                               [9, 55, 32],
                               [13, 48, 10]])
    print("First Generation:\n", new_population)
    num_generations = 500
    for generation in range(num_generations):
        fit = cal_pop_fitness(image=img, weak=wk, strong=stn, ground_truth=gnd_trt,
                              population=new_population)
        best_match_idx = np.where(fit == np.max(fit))
        print(f"Best solution in generation({generation - 1}):", new_population[best_match_idx, :])
        print("Best solution fitness:", fit[best_match_idx])
        parents = select_mating_pool(new_population, fit, PARENTS_MATING_NUMBER)
        offspring_crossover = crossover(parents, offspring_size=(
            population_size[0] - parents.shape[0], CANNY_PARAMETERS_NUMBER))
        offspring_mutation = mutation(offspring_crossover)
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

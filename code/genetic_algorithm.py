import numpy as np
from code.canny import canny, quality_assessment
from PIL import Image

CANNY_PARAMETERS_NUMBER = 4
SOLUTION_PER_POPULATION = 8
PARENTS_MATING_NUMBER = 4


def cal_pop_fitness(image, weak, strong, ground_truth, population):
    fitness = []
    for gene_i in range(population.shape[0]):
        detection = canny(image=image, weak=weak, strong=strong,
                          cutoff_frequency=population[gene_i, 0],
                          alpha=population[gene_i, 1], low=population[gene_i, 2],
                          high=population[gene_i, 3])
        true_positive_rate, false_positive_rate, accuracy = quality_assessment(
            detection=detection, ground_truth=ground_truth)
        fitness.append(true_positive_rate + accuracy - false_positive_rate)
    return np.array(fitness)


def select_mating_pool(population, fitness, num_parents):
    order = np.argsort(fit, )
    population_sorted = np.array(population, dtype=population.dtype)[order, :]
    fitness_sorted = np.array(fitness, dtype=fitness.dtype)[order]
    return population_sorted[-num_parents:], fitness_sorted[-num_parents:]


def random_take_out(population, size, probability):
    list_of_indices = list([_ for _ in range(population.shape[0])])
    probability_distribution = probability / np.sum(probability)
    draw = np.random.choice(a=list_of_indices, size=size, p=probability_distribution,
                            replace=False)
    return population[draw]


def crossover(parents, parents_fitness, offspring_size):
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
            random_value = (random_value // 2) * 2
            if offspring_crossover[idx, rnt] + random_value > 0:
                offspring_crossover[idx, rnt] = offspring_crossover[idx, rnt] + random_value
        elif rnt == 1:
            if 10 < offspring_crossover[idx, rnt] + random_value < 80:
                offspring_crossover[idx, rnt] = offspring_crossover[idx, rnt] + random_value
        elif rnt == 2:
            if offspring_crossover[idx, rnt] + random_value >= offspring_crossover[idx, rnt + 1]:
                offspring_crossover[idx, rnt + 1] += random_value
            offspring_crossover[idx, rnt] += random_value
        elif rnt == 3:
            if offspring_crossover[idx, rnt - 1] >= offspring_crossover[idx, rnt] + random_value:
                offspring_crossover[idx, rnt - 1] += random_value
            offspring_crossover[idx, rnt] += random_value
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
    new_population = []
    cutoff_normal_dist = []
    first_half_dist_for_cutoff = np.array([i for i in range(9) for _ in range(i) if i % 2 != 0])
    second_half_dist_for_cutoff = 18 - first_half_dist_for_cutoff
    dist_for_cutoff = np.concatenate((first_half_dist_for_cutoff, second_half_dist_for_cutoff))

    for _ in range(SOLUTION_PER_POPULATION):
        first_half_dist_for_cutoff = np.array(
            [i for i in range(9) for _ in range(i) if i % 2 != 0])
        second_half_dist_for_cutoff = 18 - first_half_dist_for_cutoff
        dist_for_cutoff = np.concatenate((first_half_dist_for_cutoff, second_half_dist_for_cutoff))
        cutoff = np.random.choice(dist_for_cutoff)

        alpha = np.random.randint(15, 75)
        low = np.random.randint(10, 100)
        high = np.random.randint(low + 1, 103)
        new_population.append([cutoff, alpha, low, high])
    new_population = np.array(new_population)
    print("First Generation:\n", new_population)
    num_generations = 500
    for generation in range(num_generations):
        fit = cal_pop_fitness(image=img, weak=wk, strong=stn, ground_truth=gnd_trt,
                              population=new_population)
        best_match_idx = np.where(fit == np.max(fit))
        print(f"Best solution in generation({generation}):", new_population[best_match_idx, :])
        print("Best solution fitness:", fit[best_match_idx])
        parentz, parentz_fitness = select_mating_pool(new_population, fit, PARENTS_MATING_NUMBER)
        offspring_crossover = crossover(parentz, parentz_fitness, offspring_size=(
            population_size[0] - parentz.shape[0], CANNY_PARAMETERS_NUMBER))
        offspring_mutation = mutation(offspring_crossover)
        new_population[0:parentz.shape[0], :] = parentz
        new_population[parentz.shape[0]:, :] = offspring_mutation

# -*- coding:utf-8 -*-
import numpy as np

# 假定限定条件如下
n_items = 10  # 运送货物种类
n_vehicles = 5  # 车辆总数
# 随机生成用以演示，可从excel到入数据
vehicle_capacities = np.array([5, 10, 9, 6, 7])  # 单个车辆载货量
item_weights = np.random.randint(1, 4, size=n_items)  # 每种货物的单个重量
vehicle_costs = np.random.randint(10, 20, size=n_vehicles)  # 每辆车的费用
item_demands = np.random.randint(1, 4, size=n_items)  # 每种货物的需求量
item_distances = np.random.randint(1, 11, size=n_items)  # 各个货物需要运送距离


# 遗传算法相关函数
def initialize_population(population_size, n_items, n_vehicles):
    return np.random.randint(0, n_vehicles, size=(population_size, n_items))


def evaluate_fitness(population, vehicle_capacities, item_weights, vehicle_costs, item_demands, item_distances):
    fitness_values = np.zeros(len(population))
    for i, individual in enumerate(population):
        vehicle_loads = np.zeros(n_vehicles)
        total_cost = 0
        for j in range(n_items):
            vehicle_loads[individual[j]] += item_weights[j]
            if vehicle_loads[individual[j]] > vehicle_capacities[individual[j]]:
                fitness_values[i] -= 1  # 过载，惩罚
            total_cost += vehicle_costs[individual[j]]
            total_cost += item_demands[j] * item_distances[j]
        fitness_values[i] -= total_cost
    return fitness_values


def select_parents(population, fitness_values):
    fitness_probabilities = fitness_values / sum(fitness_values)
    parent1_idx = np.random.choice(range(len(population)), p=fitness_probabilities)
    parent2_idx = np.random.choice(range(len(population)), p=fitness_probabilities)
    return population[parent1_idx], population[parent2_idx]


def crossover(parent1, parent2):
    split_point = np.random.randint(0, len(parent1))
    child1 = np.concatenate((parent1[:split_point], parent2[split_point:]))
    child2 = np.concatenate((parent2[:split_point], parent1[split_point:]))
    return child1, child2


def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            individual[i] = np.random.randint(0, n_vehicles)
    return individual


# 遗传算法参数设定
max_generations = 100  # 最大遗传代级
population_size = 50  # 每一代种群规模
mutation_rate = 0.01  # 突变概率

# Initialize population
population = initialize_population(population_size, n_items, n_vehicles)

# Evolve population
for i in range(max_generations):
    fitness_values = evaluate_fitness(population, vehicle_capacities, item_weights, vehicle_costs, item_demands,
                                      item_distances)
    parent1, parent2 = select_parents(population, fitness_values)
    child1, child2 = crossover(parent1, parent2)
    child1 = mutate(child1, mutation_rate)
    child2 = mutate(child2, mutation_rate)
    population = np.vstack((population, child1, child2))
    fitness_values = evaluate_fitness(population, vehicle_capacities, item_weights, vehicle_costs, item_demands,
                                      item_distances)
    sorted_indices = np.argsort(fitness_values)[::-1]
    population = population[sorted_indices[:population_size]]

# 演示输出
print(f"--- 基本限定条件 ---")
print(f"货物种类：{n_items}")
print(f"可用车辆数：{n_vehicles}")
print(f"--- 随机生成演示数据 ---")
print(f"每辆车辆运载能力：{vehicle_capacities}")
print(f"车辆成本：{vehicle_costs}")
print(f"货物单个重量：{item_weights}")
print(f"货物需求：{item_demands}")
print(f"货物需要被运送距离：{item_distances}")
print(f"目标：在限定条件下，最大降低运送总成本(假定出发点相同，路径相同，但距离不同)")
print(f"--- 计算最优解 ---")
best_individual = population[0]
best_fitness = evaluate_fitness(np.array([best_individual]), vehicle_capacities, item_weights,
                                vehicle_costs, item_demands, item_distances)[0]
print("最优车辆分配方案:", best_individual, "评估值", best_fitness)
print("详解：")
grouped_items = [[] for _ in range(n_vehicles)]
for i in range(n_items):
    vehicle = best_individual[i]
    grouped_items[vehicle].append(i)

total_cost = 0
for v in range(n_vehicles):
    if len(grouped_items[v]) > 0:
        cost = np.sum(item_distances[grouped_items[v]] * item_demands[grouped_items[v]])
        print(f">> 车辆号{v + 1} -- 成本:{cost}")
        print(f"装载货物号(从0数): {grouped_items[v]}")
        total_cost += cost
print(f"总成本: {total_cost}")

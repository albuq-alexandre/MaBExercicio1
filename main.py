import math
import pandas as pd
import matplotlib.pyplot as plt
import bandits
import experiment
import ratings
import random
import numpy as np


def main(num_experiments, num_iterations, num_explorations):
    """ Método principal """

    r = ratings.Ratings()
    movies_array = random.sample(r.getMovieList(), 10)
    bandits_array = list(map(lambda movie: bandits.Bandit(movie, r.getRatings(movie)), movies_array))

    rewards_array = experiment.run_greedy_bandits_experiment_with_fixed_exploration(
        bandits_array, num_experiments, num_iterations, num_explorations)

    # Calculando a recompensa acumulada
    mean_cumulative_reward = experiment.get_mean_cumulative_reward(rewards_array)

    # Fazendo o plot da expectativa real dos banditos versus a recompensa média
    # acumulada
    plt.figure(figsize=(10, 8))
    plt.plot(mean_cumulative_reward)
    legends = ['recompensa média acumulada']
    _max = max([len(x.ratings) for x in bandits_array])
    for cont, b in enumerate(bandits_array):
        bandit_real_q = (sum(b.ratings) / _max)
        plt.plot(np.ones(num_iterations) * bandit_real_q)
        legends.append(f'Filme {str(movies_array[cont])} - Q = {bandit_real_q:.4f}')
    plt.xscale('log')
    plt.xlabel('Número de iterações')
    plt.ylabel('Recompensa média acumulada')
    plt.title(f'Plot da recompensa média acumulada em {num_experiments} experimentos')
    plt.legend(legends)
    plt.show()


def mainUCB():
    nr_arms = 10
    # num_iterations = 100000
    r = ratings.Ratings()
    movies_array = random.sample(r.getMovieList(), nr_arms)
    bandits_array = list(map(lambda movie: bandits.Bandit(movie, r.getRatings(movie)), movies_array))
    num_iterations = max([len(x.ratings) for x in bandits_array])
    ads_selected, sums_of_reward, numbers_of_selections, total_reward, rewards_array = experiment.run_UCB_bandits_experiment(
        bandits_array, num_iterations, math.sqrt(2))
    # Percentual de interações que o agente teve com cada arma
    ad_hist = pd.Series(ads_selected).value_counts(normalize=True)
    # ad_hist.hist()
    df_hist = pd.DataFrame(ad_hist).reset_index()
    df_hist.rename(columns={'index': 'ad', 0: 'freq'}, inplace=True)
    # df_hist
    df_hist.plot.scatter(x='ad', y='freq', c='freq', cmap='PuBuGn')
    plt.show()

    # Fazendo o plot da expectativa real dos banditos versus a recompensa média
    mean_cumulative_reward = experiment.get_mean_cumulative_reward(rewards_array)
    # acumulada
    plt.figure(figsize=(10, 8))
    plt.plot(mean_cumulative_reward)
    legends = ['recompensa média acumulada']
    _max = num_iterations
    for cont, b in enumerate(bandits_array):
        bandit_real_q = (sum(b.ratings) / _max)
        plt.plot(np.ones(num_iterations) * bandit_real_q)
        legends.append(f'Filme {str(movies_array[cont])} - Q = {bandit_real_q:.4f}')
    plt.xscale('log')
    plt.xlabel('Número de iterações')
    plt.ylabel('Recompensa média acumulada')
    plt.title(f'Plot da recompensa média acumulada em {num_iterations} iterações')
    plt.legend(legends)
    plt.show()


# main(num_experiments=10000, num_iterations=100, num_explorations=10)
mainUCB()

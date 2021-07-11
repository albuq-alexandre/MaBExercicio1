import math
import pandas as pd
import matplotlib.pyplot as plt
import bandits
import experiment
import ratings
import random
import numpy as np


def _init(nr_arms):
    r = ratings.Ratings()
    movies_array = random.sample(r.getMovieList(), nr_arms)
    bandits_array = list(map(lambda movie: bandits.Bandit(movie, r.getRatings(movie)), movies_array))
    max_interations = max([len(x.ratings) for x in bandits_array])
    return bandits_array, movies_array, max_interations


def main(num_experiments, num_iterations, num_explorations, bandits_array, movies_array):
    """ Método greedy with fixed explorations """

    rewards_array = experiment.run_greedy_bandits_experiment_with_fixed_exploration(
        bandits_array, num_experiments, num_iterations, num_explorations)
    # Calculando a recompensa acumulada
    mean_cumulative_reward = experiment.get_mean_cumulative_reward(rewards_array)
    # Fazendo o plot da expectativa real dos banditos versus a recompensa média acumulada
    plot_comparative(bandits_array, movies_array, [mean_cumulative_reward], num_iterations, num_experiments, ['Recompensa média acumulada - Greedy'])


def main_ucb(num_experiments, num_iterations, c, seed, bandits_array, movies_array):
    nr_arms = 10
    # num_iterations = 100000
    ads_selected, sums_of_reward, numbers_of_selections, total_reward, rewards_array = experiment.run_UCB_bandits_experiment(
        bandits_array, num_iterations, c)
    # Percentual de interações que o agente teve com cada arma
    ad_hist = pd.Series(ads_selected).value_counts(normalize=True)
    # ad_hist.hist()
    df_hist = pd.DataFrame(ad_hist).reset_index()
    df_hist.rename(columns={'index': 'ad', 0: 'freq'}, inplace=True)
    # df_hist
    df_hist.plot.scatter(x='ad', y='freq', c='freq', cmap='PuBuGn')
    plt.show()
    # Calculando a recompensa acumulada
    mean_cumulative_reward = experiment.get_mean_cumulative_reward(rewards_array)
    # Fazendo o plot da expectativa real dos banditos versus a recompensa média
    plot_comparative(bandits_array, movies_array, [mean_cumulative_reward], num_iterations, num_experiments, 'Recompensa média acumulada - UCB')


def main_e(num_experiments, num_iterations, epsilon, seed, bandits_array, movies_array):
    """ Método e-greedy """
    rewards_array = experiment.run_e_greedy_bandits_experiment(
        bandits_array, num_experiments, num_iterations, epsilon, seed)
    # Calculando a recompensa acumulada
    mean_cumulative_reward = experiment.get_mean_cumulative_reward(rewards_array)
    plot_comparative(bandits_array, movies_array, [mean_cumulative_reward], num_iterations, num_experiments, 'Recompensa média acumulada - \u03B5-greedy')


def plot_comparative(bandits_array, movies_array, mean_cumulative_rewards, num_iterations, num_experiments, lbl_police):
    # Fazendo o plot da expectativa real dos banditos versus a recompensa média
    # acumulada
    legends = []
    _max = max([len(x.ratings) for x in bandits_array])
    plt.figure(figsize=(14, 8))
    for p, plot in enumerate(mean_cumulative_rewards):
        if p%2 == 0:
          plt.plot(plot, linestyle=(0, (1, 1)))
        else:
          plt.plot(plot)
        legends.append(lbl_police[p])

    for cont, b in enumerate(bandits_array):
        bandit_real_q = (sum(b.ratings) / _max)
        plt.plot(np.ones(num_iterations) * bandit_real_q, linestyle='dashed')
        legends.append(f'Filme {str(movies_array[cont])} - Q = {bandit_real_q:.4f}')
    plt.xscale('log')
    plt.xlabel('Número de iterações')
    plt.ylabel('Recompensa média acumulada')
    plt.title(f'Plot da recompensa média acumulada em {num_experiments} experimentos')
    plt.legend(legends, title = 'Qa média x real', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


def e1_e_reedy(qt_bandits, seed, num_experiments):
    epsilon = [0, 0.01, 0.1, 0.5, 0.75, 1]
    rewards_arrays = []
    mean_cumulative_rewards = []
    lbl_police = []
    bandits_array, movies_array, max_interations = _init(qt_bandits)
    for exp, eps in enumerate(epsilon):
        print(f'\u03B5-greedy com \u03B5 = {epsilon[exp]:.2f}')
        print(f'iterações = {max_interations}')
        rewards_arrays.append(
            experiment.run_e_greedy_bandits_experiment(
                bandits_array, num_experiments, max_interations, epsilon[exp], seed))
        mean_cumulative_rewards.append(
            experiment.get_mean_cumulative_reward(rewards_arrays[exp]))
        lbl_police.append(f'Recompensa média acumulada - \u03B5 = { epsilon[exp]:.2f}')
    plot_comparative(bandits_array, movies_array, mean_cumulative_rewards, max_interations, num_experiments, lbl_police)


# e1_e_reedy(5, 42, 100)

def e1_ucb(qt_bandits, num_experiments):
    c = [0, 0.5, 1, math.sqrt(2), 2, 5]
    rewards_arrays = []
    mean_cumulative_rewards = []
    lbl_police = []
    df_hists=[]
    bandits_array, movies_array, max_interations = _init(qt_bandits)
    for _c, eps in enumerate(c):
        print(f'UCB com c = {c[_c]:.2f}')
        print(f'interações = {max_interations}')
        ads_selected, sums_of_reward, numbers_of_selections, total_reward, rewards_array = \
            experiment.run_UCB_bandits_experiment(bandits_array, max_interations, c[_c])
        rewards_arrays.append(rewards_array)
        mean_cumulative_rewards.append(
            experiment.get_mean_cumulative_reward(rewards_array))
        lbl_police.append(f'Recompensa média acumulada - UCB c = { c[_c]:.2f}')
        ad_hist = pd.Series(ads_selected).value_counts(normalize=True)
        df_hist = pd.DataFrame(ad_hist).reset_index()
        df_hist.rename(columns={'index': 'ad', 0: 'freq'}, inplace=True)
        # df_hist
        df_hists.append(df_hist)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
    lin = 0
    col = 0
    for i, df in enumerate(df_hists):

        df.plot.scatter(x='ad', y='freq', c='freq', cmap='PuBuGn', ax = axes[lin, col])
        col+= 1
        if col > 2:
            lin = 1
            col = 0

    plt.show()
    plot_comparative(bandits_array, movies_array, mean_cumulative_rewards, max_interations, num_experiments, lbl_police)

e1_ucb(10, 50)

import matplotlib.pyplot as plt
import Bandits
import Experiment
import Ratings
import random
import numpy as np


def main(num_experiments, num_iterations, num_explorations):
    """ Método principal """

    r = Ratings.Ratings()
    movies_array = random.sample(r.getMovieList(), 10)
    bandits_array = list(map(lambda movie: Bandits.Bandit(r.getRatings(movie)), movies_array))

    rewards_array = Experiment.run_greedy_bandits_experiment_with_fixed_exploration(
        bandits_array, num_experiments, num_iterations, num_explorations)

    # Calculando a recompensa acumulada
    mean_cumulative_reward = Experiment.get_mean_cumulative_reward(rewards_array)

    # Fazendo o plot da expectativa real dos banditos versus a recompensa média
    # acumulada
    plt.figure(figsize=(10, 8))
    plt.plot(mean_cumulative_reward)
    legends = ['recompensa média acumulada']
    _max = max ([ len(x.ratings) for x in bandits_array])
    for cont, b in enumerate(bandits_array):
        plt.plot(np.ones(num_iterations) * (sum(b.ratings)/_max))
        legends.append('Filme ' + str(movies_array[cont]))
    plt.xscale('log')
    plt.xlabel('Número de iterações')
    plt.ylabel('Recompensa média acumulada')
    plt.title(f'Plot da recompensa média acumulada em {num_experiments} experimentos')
    plt.legend(legends)
    plt.show()


main(num_experiments=10000, num_iterations=100, num_explorations=10)

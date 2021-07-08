import numpy as np
import random
import math
from numpy.random.mtrand import RandomState


def greedy_policy(bandits):
    """ Função responsável por escolher o bandido segundo uma estratégia greedy.
      Retorna o bandido correspondente à greedy action.
  """
    Q_values = [b.Q_t for b in bandits]
    greedy_index = np.argmax(Q_values)
    greedy_bandit = bandits[greedy_index]
    return greedy_bandit, greedy_index


def random_policy(bandits, seed=42):
    """ Função responsável por escolher o bandido segundo uma estratégia randômica.
      Retorna o bandido correspondente à random action.
  """
    random.seed(seed)
    rnd_index = random.choice(range(len(bandits)))
    rnd_bandit = bandits[rnd_index]
    return rnd_bandit, rnd_index


def e_greedy_policy(bandits, epsilon=0.1, seed=42):
    """ Função responsável por escolher o bandido segundo uma estratégia epslon-greedy.
      Retorna o bandido correspondente à greedy action.
  """
    Q_values = [b.Q_t for b in bandits]
    if RandomState(seed).choice([True, False], p=[epsilon, 1.0 - epsilon]):
        e_greedy_index = random.choice(range(len(bandits)))
    else:
        e_greedy_index = np.argmax(Q_values)

    e_greedy_bandit = bandits[e_greedy_index]

    return e_greedy_bandit, e_greedy_index


def UCB_policy(bandits):
    """ Função responsável por escolher o bandido segundo uma estratégia UCB.
      Retorna o bandido correspondente à UCB action.
  """
    num_ads = len(bandits)
    ads_selected = []
    total_reward = 0
    numbers_of_selections = [0] * num_ads
    sums_of_reward = [0] * num_ads
    total_reward = 0

    for i in range(0, len(bandits)):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_reward[i] / numbers_of_selections[i]
            delta_i = math.sqrt(2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
    else:
        upper_bound = 1e400
    if upper_bound > max_upper_bound:
        max_upper_bound = upper_bound
        ad = i

        ads_selected.append(ad)
        numbers_of_selections[ad] += 1
        reward = dataset.values[n, ad]
        sums_of_reward[ad] += reward
        total_reward += reward

    return rnd_bandit, rnd_index


def interact_with_bandit(bandit):
    """ Função que interage com um bandido. A interação envolve a coleta da
      recompensa e a atualização da média estimada.
  """
    # Coleta a recompensa da interação
    reward = bandit.pull()

    # Atualiza a estimação da média
    bandit.update(reward)

    return reward


def fixed_exploration(bandits, num_explorations):
    """ Método que faz um número fixo de interações com cada bandido.
  """
    for b in bandits:
        for n in range(num_explorations):
            # Nessa etapa inicial de exploração, os rewards não serão armazenados.
            interact_with_bandit(b)


def run_greedy_bandits_experiment_with_fixed_exploration(
        bandits_array, num_experiments, greedy_iterations, num_explorations):
    """ Simula um cenário dos banditos com a escolha greedy de ação """

    rewards_array = np.zeros(greedy_iterations)

    for e in np.arange(num_experiments):
        print(f'Experimento {e}', end='\r')
        list(map(lambda b: b.reset(), bandits_array))
        # list(map(lambda b: print(b.Q_t, b.t), bandits_array))

        # Faz a exploração inicial para estimar um valor inicial para Q_t
        fixed_exploration(bandits_array, num_explorations)

        # list(map(lambda b: print(b.Q_t, b.t), bandits_array))

        for n in np.arange(greedy_iterations):
            # Sempre há escolha da ação greedy.
            greedy_bandit, greedy_index = greedy_policy(bandits_array)

            # Realiza a interação com o bandido que corresponde à ação greedy
            curr_reward = interact_with_bandit(greedy_bandit)

            rewards_array[n] += curr_reward
        # list(map(lambda b: print(b.Q_t, b.t), bandits_array))

    rewards_array /= num_experiments

    return rewards_array


def get_mean_cumulative_reward(reward_array):
    cumulative_reward = np.cumsum(reward_array)
    mean_cumulative_reward = cumulative_reward / (np.arange(len(reward_array)) + 1)
    return mean_cumulative_reward


def run_UCB_bandits_experiment(bandits_array, ucb_iterations, c):
    num_iterations = ucb_iterations
    num_ads = len(bandits_array)
    ads_selected = []
    reward_array = []
    numbers_of_selections = [0] * num_ads
    sums_of_reward = [0] * num_ads
    total_reward = 0

    for n in range(0, num_iterations):
        ad = 0
        max_upper_bound = 0
        for i in range(0, num_ads):
            if (numbers_of_selections[i] > 0):
                average_reward = sums_of_reward[i] / numbers_of_selections[i]
                delta_i = c * math.sqrt(math.log(n + 1) / numbers_of_selections[i])
                upper_bound = average_reward + delta_i
            else:
                upper_bound = 1e400
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                ad = i
        ads_selected.append(bandits_array[ad].movieId)
        numbers_of_selections[ad] += 1
        reward = bandits_array[ad].pull()
        reward_array.append(reward)
        sums_of_reward[ad] += reward
        total_reward += reward
    print('Recompensa total', total_reward)
    print('Número de seleções', numbers_of_selections)
    print('Recompensas por arma', sums_of_reward)
    print()
    return ads_selected, sums_of_reward, numbers_of_selections, total_reward, reward_array


def run_e_greedy_bandits_experiment(
        bandits_array, num_experiments, greedy_iterations, epsilon, seed):
    """ Simula um cenário dos banditos com a escolha e-greedy de ação """

    rewards_array = np.zeros(greedy_iterations)

    for e in np.arange(num_experiments):
        print(f'Experimento {e}', end='\r')
        list(map(lambda b: b.reset(), bandits_array))
        # list(map(lambda b: print(b.Q_t, b.t, end="\r"), bandits_array))
        # print('\n')

        for n in np.arange(greedy_iterations):
            # Sempre há escolha da ação greedy.
            greedy_bandit, greedy_index = e_greedy_policy(bandits_array, epsilon, seed)

            # Realiza a interação com o bandido que corresponde à ação greedy
            curr_reward = interact_with_bandit(greedy_bandit)

            rewards_array[n] += curr_reward
        # list(map(lambda b: print(b.Q_t, b.t, end="\r"), bandits_array))

    rewards_array /= num_experiments

    return rewards_array


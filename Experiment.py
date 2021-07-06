import numpy as np


def greedy_policy(bandits):
    """ Função responsável por escolher o bandido segundo uma estratégia greedy.
      Retorna o bandido correspondente à greedy action.
  """
    Q_values = [b.Q_t for b in bandits]
    greedy_index = np.argmax(Q_values)
    greedy_bandit = bandits[greedy_index]
    return greedy_bandit, greedy_index


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

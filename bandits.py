class Bandit:
    """ Classe que define um bandido com recompensa seguindo uma distribuição
      uniforme
  """

    def __init__(self, movieId, movie_ratings: list):
        """ Inicia a classe com a distribuição de recompensas baseadas nas avaliacoes do filme.
    """
        self.ratings = movie_ratings
        self.Q_t = 0
        self.t = 0
        self.interact = 0
        self.movieId = movieId

    def pull(self):
        """ Método que define a interação com o bandido. A recompensa é resultante
        de uma distribuição uniforme de média especificada.
    """
        if self.interact >= len(self.ratings):
            _reward = 0
        else:
            _reward = self.ratings[self.interact]
        self.interact += 1
        return _reward

    def update(self, curr_reward):
        """ Método responsável por atualizar a estimação da expectativa da
        recompensa
    """
        self.t += 1
        self.Q_t = self.Q_t + (1.0 / self.t) * (curr_reward - self.Q_t)

    def reset(self):
        """ Método que limpa as estimações prévias de Q_t"""
        self.t = 0
        self.Q_t = 0
        self.interact = 0

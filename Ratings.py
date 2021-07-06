import numpy as np
import pandas as pd

class Ratings:
    """ Classe que carrega o dataset e constroi uma distribuição baseada no rating do filme
  """
    def __init__(self):
        # Carregando os ratings dos filmes
        self.df_ratings = pd.read_csv('./ml-25m/ratings.csv')
        # Transformando as notas 4 e 5 = 1 ; notas 0 a 3 = 0
        self.df_ratings['tumbsup'] = [1 if x >= 4 else 0 for x in self.df_ratings['rating']]

        # Excluindo os filmes com menos de 1000 avaliações
        self.s = self.df_ratings.groupby('movieId')[['movieId']].size().reset_index()
        self.s.rename(columns={0: 'movieCount'}, inplace=True)
        self.df_ratings = self.df_ratings.merge(self.s, how='inner', on='movieId')
        self.df_ratings.drop(self.df_ratings[self.df_ratings['movieCount'] < 10000].index, inplace=True)

        # Calculo da probabilidade de cada filme
        self.s = self.df_ratings.groupby(['movieId'])['tumbsup'].agg('sum').reset_index()
        self.s.rename(columns={'tumbsup': 'tumbsCount'}, inplace=True)
        self.df_ratings = self.df_ratings.merge(self.s, how='inner', on='movieId')
        self.df_ratings['movieProb'] = self.df_ratings['tumbsCount'] / np.max(self.df_ratings['movieCount'])

        # dropando as colunas para economizar memória
        self.df_ratings = self.df_ratings[{'userId', 'movieId', 'tumbsup', 'movieCount', 'tumbsCount', 'movieProb'}]
        self.s = None

    def getRatings(self, r_movieId):
        df = self.df_ratings.loc[self.df_ratings['movieId'] == r_movieId]
        df = df[{'movieId', 'tumbsup'}]
        _movieRatingsList: list = df.pivot(columns="movieId", values="tumbsup")[r_movieId].tolist()
        return _movieRatingsList

    def getMovieList(self):
        return self.df_ratings['movieId'].unique().tolist()


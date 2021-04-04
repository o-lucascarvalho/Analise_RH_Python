# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 16:22:57 2021

@author: Lucas
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

dadosRH = pd.read_csv('dadosRH.csv')
dadosRH.head()
dadosRH.shape

dadosRH.isnull().sum()
dadosRH.groupby(['educacao']).count()
sns.countplot(dadosRH['educacao'])
dadosRH.groupby(['aval_ano_anterior']).count()
sns.countplot(dadosRH['aval_ano_anterior'])
dadosRH['educacao'].fillna(dadosRH['educacao'].mode()[0], inplace = True)
dadosRH['aval_ano_anterior'].fillna(dadosRH['aval_ano_anterior'].median(), inplace = True)
dadosRH.isnull().sum()
dadosRH.shape
dadosRH.groupby(['educacao']).count()
dadosRH.groupby(['aval_ano_anterior']).count()
dadosRH.groupby(['promovido']).count()
sns.countplot(dadosRH['promovido'])
df_classe_majoritaria = dadosRH[dadosRH.promovido==0]
df_classe_minoritaria = dadosRH[dadosRH.promovido==1]
df_classe_majoritaria.shape
# Upsample da classe minoritária
from sklearn.utils import resample
df_classe_minoritaria_upsampled = resample(df_classe_minoritaria, 
                                           replace = True,     
                                           n_samples = 50140,   
                                           random_state = 150) 
dadosRH_balanceados = pd.concat([df_classe_majoritaria, df_classe_minoritaria_upsampled])
dadosRH_balanceados.promovido.value_counts()
dadosRH_balanceados.info()
sns.countplot(dadosRH_balanceados['promovido'])
dadosRH_balanceados.to_csv('dadosRH_modificado.csv', encoding = 'utf-8', index = False)

dataset = pd.read_csv('dadosRH_modificado.csv')
dataset.head()

#Correlação entre atributos dos dados
corr = dataset.corr()
sns.heatmap(corr, cmap = "PuRd", linewidths = 0.1)
plt.show()

#Tempo de serviço dos funcionários
sns.distplot(dataset['tempo_servico'], color = 'green')
plt.title('Distribuição do Tempo de Serviço dos Funcionários', fontsize = 15)
plt.xlabel('Tempo de Serviço em Anos', fontsize = 15)
plt.ylabel('Total')
plt.show()

#Distribuição das avaliações do ano anterior
dataset['aval_ano_anterior'].value_counts().sort_values().plot.bar(color = 'blue', figsize = (10, 5))
plt.title('Distribuição da Avaliação do Ano Anterior dos Funcionários', fontsize = 15)
plt.xlabel('Avaliações', fontsize = 15)
plt.ylabel('Total')
plt.show()

#Distribuição da idade dos funcionários
sns.distplot(dataset['idade'], color = 'darkblue')
plt.title('Distribuição da Idade dos Funcionários', fontsize = 15)
plt.xlabel('Idade', fontsize = 15)
plt.ylabel('Total')
plt.show()

#Distribuição de treinamentos realizados
sns.violinplot(dataset['numero_treinamentos'], color = 'darkgray')
plt.title('Número de Treinamentos Feitos Pelos Funcionários', fontsize = 15)
plt.xlabel('Número de Treinamentos', fontsize = 15)
plt.ylabel('Frequência')
plt.show()

#Proporção de funcionários por canal de recrutamento
dataset['canal_recrutamento'].value_counts()
fatias = [55375, 42358, 2547]
labels = "Outro", "Outsourcing", "Indicação"
colors = ['gray', 'orange', 'yellow']
explode = [0.1, 0.1, 0.1]
plt.pie(fatias, labels = labels, colors = colors, explode = explode, shadow = True, autopct = "%.2f%%")
plt.title('Percentual de Funcionários Por Canal de Recrutamento', fontsize = 15)
plt.axis('off')
plt.show()

#Relação entre promoção e avaliação de anos anteriores
data = pd.crosstab(dataset['aval_ano_anterior'], dataset['promovido'])
data.div(data.sum(1).astype(float), axis = 0).plot(kind = 'bar', 
                                                   stacked = True, 
                                                   figsize = (16, 9), 
                                                   color = ['darkred', 'darkgreen'])
plt.title('Relação Entre Avaliação do Ano Anterior e a Promoção', fontsize = 15)
plt.xlabel('Avaliação do Ano Anterior', fontsize = 15)
plt.legend()
plt.show()























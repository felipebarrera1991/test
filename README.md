# Data Science Teste

Para desenvolver este teste foi usada a versão completa dos dados (**bank-full.csv**) da [fonte oficial](https://archive.ics.uci.edu/ml/datasets/bank+marketing) fornecida no documento.

Foi usada a versão Python 3.6.5 com Anaconda, Inc. Os frameworks usados para o desenvolvimento do trabalho foram:

1. [Pandas](https://pandas.pydata.org/): estruturas de dados de alto desempenho.
2. [Seaborn](https://seaborn.pydata.org/):visualização de dados Python baseada no matplotlib.
3. [Numpy](http://www.numpy.org/): computação científica.
4. [Matplotlib](https://matplotlib.org/): visualização de dados.

## Qual profissão tem mais tendência a fazer um empréstimo? De qual tipo?

Inicialmente foram importados os dados usando a função pd.read_cvs() e foram verificadas as dimensões do dataframe

```python
data = pd.read_csv('bank-full.csv', sep=';', quotechar='"', encoding='utf8',parse_dates=True)
data.shape
```

Logo, foram identificadas as variáveis asignadas aos empréstimos (housing e loan) e as profissões (job). É verificado que as variáveis são do tipo object e devem ser transformadas em variáveis numéricas. Foi criado um novo dataframe e agrupado pela profissão.

```python
new_df = pd.get_dummies(data[['job', 'housing', 'loan']], columns=['housing', 'loan'])
new_df_group = new_df.groupby('job', as_index=False).sum()
```

Determina-se a relação dentre as variáveis dos empréstimos e os clientes.

```python
new_df_group['housing'] = new_df_group['housing_yes']/(new_df_group['housing_yes'] + new_df_group['housing_no'])
new_df_group['loan'] = new_df_group['loan_yes']/(new_df_group['loan_yes'] + new_df_group['loan_no'])
```

São organizados os dados de maior a menor em função da variável housing e finalmente é graficada a resposta.

```python
new_df_group = new_df_group.sort_values(by='housing', ascending=False)
```

![question_1](https://user-images.githubusercontent.com/28451312/46163286-27ca8480-c261-11e8-933e-a6676b7128bc.png)

Da figura apresentada, pode-se observar que a profissão com uma maior tendência a fazer empréstimo é blue-collar






![question_2](https://user-images.githubusercontent.com/28451312/46163326-44ff5300-c261-11e8-9c5d-3e47ad509971.png)




![question_3](https://user-images.githubusercontent.com/28451312/46163367-652f1200-c261-11e8-9112-71590bfc0f08.png)




![question_4](https://user-images.githubusercontent.com/28451312/46163415-87c12b00-c261-11e8-84af-148d98a65a34.png)





![question_5](https://user-images.githubusercontent.com/28451312/46163438-a4f5f980-c261-11e8-8a8b-ff3f675dab6a.png)



![question_6](https://user-images.githubusercontent.com/28451312/46163499-bb03ba00-c261-11e8-84c0-4f2b6851df1d.png)



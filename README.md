# Data Science Teste

Para desenvolver este teste foi usada a versão completa dos dados (**bank-full.csv**) da [fonte oficial](https://archive.ics.uci.edu/ml/datasets/bank+marketing) fornecida no documento.

Foi usada a versão Python 3.6.5 com Anaconda, Inc. Os frameworks usados para o desenvolvimento do trabalho foram:

1. [Pandas](https://pandas.pydata.org/): estruturas de dados de alto desempenho.
2. [Seaborn](https://seaborn.pydata.org/):visualização de dados Python baseada no matplotlib.
3. [Numpy](http://www.numpy.org/): computação científica.
4. [Matplotlib](https://matplotlib.org/): visualização de dados.

# Solução

### Qual profissão tem mais tendência a fazer um empréstimo? De qual tipo?

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

Da figura apresentada, pode-se observar que a profissão com uma maior tendência a fazer empréstimo é blue-collar.

### Fazendo uma relação entre número de contatos e sucesso da campanha quais são os pontos relevantes a serem observados?

Inicialmente foram identificadas as variáveis de sucesso da campanha (y) e o número de contatos (campaign). Logo, a variável de sucesso de campanha (y) foi agrupada e determinado o número de ocorrências en todo o conjunto de dados.

```python
new_df2 = data.groupby('y', as_index=False).count()
```

Logo, foi usada a função barplot de [Seaborn](https://seaborn.pydata.org/) para determinar o número de contatos durante a campanha divido en dois grupos, que teve sucesso e que não teve sucesso. Foi feito também um gráfico circular para que a informação resultante fosse mais clara. 

```python
f, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
sns.barplot(x='campaign', y='y', data=new_df2, ax=ax1)
ax1.set(xlim=(0, new_df2['campaign'].max()*1.2), ylabel='Client Subscribed', xlabel='Campaing')
sns.despine(left=True, bottom=True)
# Pie plot
explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
ax2.pie(new_df2['campaign'], explode=explode, autopct='%1.1f%%',shadow=True, startangle=90)
ax2.axis('equal')
plt.tight_layout()
```
Da figura, pode-se observar que a campanha teve só um 11.7% de successo na adesão (aproximadamente 5300 pessoas).

![question_2](https://user-images.githubusercontent.com/28451312/46163326-44ff5300-c261-11e8-9c5d-3e47ad509971.png)

### Baseando-se nos resultados de adesão desta campanha qual o número médio e o máximo de ligações que você indica para otimizar a adesão?

Para determinar o número médio e máximo de ligações, foi feito um diagrama de caixa. Este diagrama de caixa é uma ferramenta visual que representa variações de dados por meio de quartis. Para fazer este diagrama, foi re-definido um dataframe com as descripções do número de contatos realizados durante a campanha ao cliente, embora este tivesse aceito ou não a adesão.

```python
new_df3 = pd.DataFrame({'campaign_no': data[data['y'] == 'no']['campaign'].describe()})
new_df3['campaign_yes'] = data[data['y'] == 'yes']['campaign'].describe()
```
Com a descripção obtida, foi graficado o diagrama de caixa como apresentado a seguir.

![question_3](https://user-images.githubusercontent.com/28451312/46163367-652f1200-c261-11e8-9112-71590bfc0f08.png)

Deste diagrama observa-se que o número médio de ligações para obter a adesão foram duas, e o recomendado para evitar clientes discrepantes (dispersão de puntos fora da caixa) é dado pelo límite superior da caixa, neste caso, 6 chamadas. 

### O resultado da campanha anterior tem relevância na campanha atual?

Para determinar se o resultado da campanha anterior teve relevância na campanha atual, inicialmente, foi identificada a variável do resultado da campanha de marketing anterior (poutcome) e a variável de sucesso da campanha (y). Logo, foi feita uma correlação entre estas variáveis tentando achar releância entre os resultados tanto positivos como negativos. O método usado para a correlação foi o método de Pearson através da função .corr().

```python
new_df4 = pd.get_dummies(data[['poutcome', 'y']])
correlation = new_df4.corr(method='pearson')
```

Uma vez determinada a correlação, é obtida uma matriz de confusão ou Heatmap através do framework [Seaborn](https://seaborn.pydata.org/).

```python
mask = np.zeros_like(correlation)
mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):
    plt.figure(figsize=(12,10))
    ax = sns.heatmap(correlation, mask=mask, vmax=.4, square=True,annot=True, cmap='YlGnBu')
    plt.savefig('question_4.png')
```

![question_4](https://user-images.githubusercontent.com/28451312/46163415-87c12b00-c261-11e8-84af-148d98a65a34.png)

Pode-se verificar que existe um coeficiente de correlação de 0.31 entre o sucesso da campanha anterior e a campanha atual.

### Qual o fator determinante para que o banco exija um seguro de crédito?

Inicialmente deve ser identificado se existe ou não um default no cliente, para logo depois, validar o saldo médio anual (em euros) que é o fator determinante para que o banco exija o seguro do crédito. 

```python
default_y = data[(data['default'] == 'yes') ]['balance']
default_n = data[(data['default'] == 'no') ]['balance']

new_df5 = pd.DataFrame({'default_yes': data[(data['default'] == 'yes') ]['balance'].describe()})
new_df5['default_no'] = data[(data['default'] == 'no') ]['balance'].describe()
```

Usa-se a função distplot() do framework [Seaborn](https://seaborn.pydata.org/) para obter a distribuição do balance dos clientes com default e sem default.

```python
f, (ax1, ax2) = plt.subplots(1,2,figsize=(10,6))
sns.distplot(default_y, kde=True, bins=20, ax=ax1)
ax1.set(ylabel='Density', xlabel='Balance for Default')
sns.despine(left=True, bottom=True)

sns.distplot(default_n, kde=True, bins=20, ax=ax2)
ax2.set(ylabel='Density', xlabel='Balance for No Default')
plt.tight_layout()
plt.savefig('question_5.png')
```

![question_5](https://user-images.githubusercontent.com/28451312/46163438-a4f5f980-c261-11e8-8a8b-ff3f675dab6a.png)

Na Figura obtida, observa-se que o banco exijirá um seguro de crédito para o cliente que tem defaul, enquanto que para os clientes que não apresentam default, o banco não exijirá seguro.

### Quais são as características mais proeminentes de um cliente que possua empréstimo imobiliário?

Inicialmente foram definidas as variáveis do tipo object como as características a serem usadas no análisis. Estas características são:

```python
Caracteristics = ['job', "marital", "education", "default", "loan"]
```
Uma vez identificadas estas características, foram obtidos os clientes com empréstimo imobiliário (housing).

```python
new_df6 = data[data['housing'] == 'yes'][Caracteristics].describe()
```

Assim, foi obtido que, a característica mais proeminante de um cliente que possua empréstimo imobiliário são:

```python
fig = plt.figure(figsize=(6,4))
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
ax.bar(Caracteristics,new_df6.loc['freq'])
ax.set_ylabel('Frequency')
ax.set_xlabel('Characteristics')
plt.savefig('question_6.png')
```

1. sem (default).
2. sem empréstimo (loan).
3. Casado (Marial).
4. Ensino superior (Education)
5. blue-collar (job).

![question_6](https://user-images.githubusercontent.com/28451312/46163499-bb03ba00-c261-11e8-84c0-4f2b6851df1d.png)


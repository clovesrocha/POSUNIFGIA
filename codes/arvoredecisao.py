#Exemplo árvore decisão com Python e sklearn
#Prof. Cloves Rocha IA e AD
 
from sklearn import tree

features = [[100, 1], [150, 1],
           [50, 0], [80, 0]]
labels = [0, 0, 1, 1] # 0 é CONTRATATE! e 1 é NAO CONTRATE

# o classificador encontra padrões nos dados de treinamento
clf = tree.DecisionTreeClassifier() # instância do classificador 
clf = clf.fit(features, labels) # fit encontra padrões nos dados

# iremos utilizar para classificar UM CANDIDATO A VAGA DE CIENTISTA DE DADOS
print(clf.predict([[80, 0]]))


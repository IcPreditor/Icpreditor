##Import comuns
import numpy as np
import pandas as pd
##Import dados processados
import sys
sys.path.insert(1, r'processing')
import processing
##Import Divisor de teste treino
from sklearn.model_selection import train_test_split
##Import Modelo Regressão Logística 
from sklearn.linear_model import LogisticRegression
##Import Metrics para matriz de confusão
from sklearn import metrics
##Import pyplot para plot da matriz de confusão
import matplotlib.pyplot as plt
import seaborn as sns
##Import para classificação da matriz de confusão
from sklearn.metrics import classification_report

## Data
X_train, Y_train, X_test, Y_test,dataframe,feature_cols = processing.getInputOutput(undersampling=False,regressao=True)
logreg = LogisticRegression(random_state=16,max_iter=100000)
logreg.fit(X_train, Y_train)

print("Coeficiente de Regressão das Variáveis:")
coeficientes = logreg.coef_[0]
coeficientes_dict = {}
for i in range(len(feature_cols)):
    coeficientes_dict[feature_cols[i]]=coeficientes[i]
for item in sorted(coeficientes_dict,key=coeficientes_dict.get):
    print(f"{item} - [{coeficientes_dict[item]:.5f}]")

Y_pred = logreg.predict(X_test)


# Exibir previsões feitas pelo modelo
# Matriz de confusão 
cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)


#Plot da matrix
class_names=['Não evasão','Evasão'] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Valor Real')
plt.xlabel('Previsão')
plt.show()

#Métricas de avliação matriz de confusão
target_names = ["Não Evasão","Evasão"]
print(classification_report(Y_test,Y_pred,target_names=target_names))

#Curva ROC
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba,pos_label='1')
auc = metrics.roc_auc_score(Y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()





# Exibir coeficientes de cada variável
print("Coeficiente de Regressão das Variáveis:")

# Calcular e organizar coeficientes em um DataFrame para melhor análise
coef_df = pd.DataFrame({
    'Variável': feature_cols,
    'Coeficiente': logreg.coef_[0]
})

# Ordenar pelo valor absoluto do coeficiente para identificar as variáveis com maior impacto
coef_df['Impacto Absoluto'] = coef_df['Coeficiente'].abs()
coef_df = coef_df.sort_values(by='Impacto Absoluto', ascending=False)

# Exibir as variáveis em ordem decrescente de impacto
print(coef_df[['Variável', 'Coeficiente']])

# Visualizar os coeficientes em um gráfico de barras
plt.figure(figsize=(10, 6))
sns.barplot(data=coef_df, y='Variável', x='Coeficiente', hue='Coeficiente', dodge=False, palette='coolwarm', legend=False)
plt.title('Impacto das Variáveis na Evasão')
plt.xlabel('Coeficiente')
plt.ylabel('Variável')
plt.axvline(0, color='black', linewidth=0.5)  # Linha para separar coeficientes positivos e negativos
plt.show()


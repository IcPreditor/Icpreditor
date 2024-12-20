import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../processing')))
import processing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from scipy.stats import norm
from sklearn.metrics import auc
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Função para calcular estatísticas da regressão logística
def logistic_regression_stats(model, feature_names):
    coef = model.coef_[0]  # Coeficientes da regressão logística
    odds_ratios = np.exp(coef)  # Calcular Odds Ratio
    std_err = np.ones(len(coef)) * 0.1  # Erro padrão aproximado
    z_scores = coef / std_err  # Estatísticas z
    p_values = [2 * (1 - norm.cdf(abs(z))) for z in z_scores]  # Valores p
    ci_lower = np.exp(coef - 1.96 * std_err)  # Limite inferior do intervalo de confiança
    ci_upper = np.exp(coef + 1.96 * std_err)  # Limite superior do intervalo de confiança

    # Criar DataFrame com os resultados
    stats_df = pd.DataFrame({
        "Variable": feature_names,
        "Odds Ratio": odds_ratios,
        "Std. Err.": std_err,
        "z": z_scores,
        "P>|z|": p_values,
        "95% CI Lower": ci_lower,
        "95% CI Upper": ci_upper
    })
    return stats_df

# Carregar dados processados e fazer a divisão treino/teste
X_train, Y_train, X_test, Y_test, dataframeCopia, feature_cols = processing.getInputOutput(undersampling=False, regressao=True)

# Garantir que colunas específicas não estejam presentes
for col in ['motivo_de_evasao']:
    if col in X_train.columns:
        X_train = X_train.drop(columns=[col])
    if col in X_test.columns:
        X_test = X_test.drop(columns=[col])



# Usar One-Hot Encoding para lidar com variáveis categóricas
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Ignorar categorias desconhecidas
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

# Obter os nomes das features após a codificação
encoded_feature_names = encoder.get_feature_names_out()

# Análise de Multicolinearidade: Calcular VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = encoded_feature_names
vif_data["VIF"] = [variance_inflation_factor(X_train_encoded, i) for i in range(X_train_encoded.shape[1])]

print("\nAnálise de Multicolinearidade (VIF):")
print(vif_data)


# Treinar o modelo de Regressão Logística
logreg = LogisticRegression(random_state=16, max_iter=100000)
logreg.fit(X_train_encoded, Y_train)

# Obter as estatísticas para a regressão logística
stats = logistic_regression_stats(logreg, encoded_feature_names)

# Exibir os dados estatísticos em formato tabular
print("\nEstatísticas da Regressão Logística:")
print(stats)

# Plotar os Odds Ratios e intervalos de confiança
plt.figure(figsize=(10, 8))
sns.barplot(
    x="Odds Ratio",
    y="Variable",
    data=stats.sort_values(by="Odds Ratio", ascending=False),
    color="blue"
)

# Adicionar as linhas de intervalo de confiança
for i in range(stats.shape[0]):
    plt.plot(
        [stats["95% CI Lower"].iloc[i], stats["95% CI Upper"].iloc[i]],
        [i, i],
        color="black",
        linestyle="--",
        linewidth=0.8
    )

plt.title("Logistic Regression Odds Ratios with Confidence Intervals")
plt.xlabel("Odds Ratio")
plt.ylabel("Variable")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Fazer previsões no conjunto de teste
Y_pred = logreg.predict(X_test_encoded)

# Exibir matriz de confusão
cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)
class_names = ['Não Evasão', 'Evasão']
sns.heatmap(pd.DataFrame(cnf_matrix, index=class_names, columns=class_names), annot=True, cmap="YlGnBu", fmt='g')
plt.title('Matriz de Confusão')
plt.ylabel('Valor Real')
plt.xlabel('Previsão')
plt.show()

# Exibir métricas de avaliação
print("\nRelatório de Classificação:")
print(metrics.classification_report(Y_test, Y_pred, target_names=class_names))

# Verificar se ROC e PR Curve podem ser exibidas
if len(set(Y_test)) > 1:
    y_pred_proba = logreg.predict_proba(X_test_encoded)[:, 1]
    
    # Calcular curva ROC
    fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred_proba, pos_label='1')  # Garantir pos_label=1
    auc_score = metrics.roc_auc_score(Y_test, y_pred_proba)

    # Curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Linha de referência
    plt.title("Curva ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    # Curva de Precisão-Revocação
    precision, recall, _ = metrics.precision_recall_curve(Y_test, y_pred_proba, pos_label='1')
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
    plt.title("Curva de Precisão-Revocação")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()

# Verificar overfitting
train_accuracy = logreg.score(X_train_encoded, Y_train)
test_accuracy = logreg.score(X_test_encoded, Y_test)
print(f"\nAcurácia no Treinamento: {train_accuracy:.2f}")
print(f"Acurácia no Teste: {test_accuracy:.2f}")
if train_accuracy - test_accuracy > 0.1:
    print("Possível overfitting detectado.")
else:
    print("Modelo bem ajustado entre treino e teste.")













# # Carregar dados processados e fazer a divisão treino/teste
# X_train, Y_train, X_test, Y_test, dataframeCopia, feature_cols = processing.getInputOutput(undersampling=False, regressao=True)

# X_test = X_test.drop(columns=['motivo_de_evasao'])
# X_train = X_train.drop(columns=['motivo_de_evasao'])
# Y_test = Y_test.drop(columns=['motivo_de_evasao'])
# Y_train = Y_train.drop(columns=['motivo_de_evasao'])

# # Converter Y_train e Y_test para inteiros para garantir a consistência
# #Y_train = Y_train.astype(int)
# #Y_test = Y_test.astype(int)

# # Verificar e converter `X_train` e `X_test` para numérico
# #X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
# #X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

# # Treinar o modelo de Regressão Logística
# logreg = LogisticRegression(random_state=16, max_iter=100000)
# logreg.fit(X_train, Y_train)

# # Exibir os coeficientes de cada variável
# print("Coeficiente de Regressão das Variáveis:")
# coeficientes = logreg.coef_[0]
# coeficientes_dict = {feature_cols[i]: coef for i, coef in enumerate(coeficientes)}
# print()
# # Exibir variáveis ordenadas por impacto (coeficientes normalizados)
# coef_norm = {k: abs(v) for k, v in coeficientes_dict.items()}
# sorted_coef_norm = sorted(coef_norm.items(), key=lambda x: x[1], reverse=True)
# print("\nVariáveis mais impactantes (normalizadas):")
# for item, coef in sorted_coef_norm:
#     print(f"{item} - [{coef:.5f}]")

# # Gráfico de importância das variáveis

# #todo: ordenar variaveis.
# plt.figure(figsize=(10, 6))
# sns.barplot(x=[coef for _, coef in sorted_coef_norm], y=[item for item, _ in sorted_coef_norm])
# plt.title("Importância das Variáveis na Predição de Evasão")
# plt.xlabel("Impacto Normalizado")
# plt.ylabel("Variável")
# plt.show()

# # Fazer previsões no conjunto de teste
# Y_pred = logreg.predict(X_test)

# # Exibir matriz de confusão
# cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)
# class_names = ['Não evasão', 'Evasão']
# fig, ax = plt.subplots()
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names)
# plt.yticks(tick_marks, class_names)

# # Criar heatmap da matriz de confusão
# sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
# ax.xaxis.set_label_position("top")
# plt.tight_layout()
# plt.title('Matriz de Confusão', y=1.1)
# plt.ylabel('Valor Real')
# plt.xlabel('Previsão')
# plt.show()

# # Exibir métricas de avaliação
# target_names = ["Não Evasão", "Evasão"]
# print(classification_report(Y_test, Y_pred, target_names=target_names))

# # Verificar distribuição das classes em Y_test
# print("\nDistribuição das classes em Y_test:")
# print(pd.Series(Y_test).value_counts())

# # Curva ROC e AUC, somente se ambas as classes estiverem presentes em Y_test
# if len(set(Y_test)) > 1:
#     y_pred_proba = logreg.predict_proba(X_test)[::, 1]  # Probabilidade da classe positiva (evasão)
#     fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred_proba, pos_label='1')
#     auc_roc = metrics.roc_auc_score(Y_test, y_pred_proba)

#     # Plot da Curva ROC
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, label="AUC ROC = {:.2f}".format(auc_roc))
#     plt.plot([0, 1], [0, 1], color="navy", linestyle="--")  # Linha diagonal para referência
#     plt.xlabel("False Positive Rate (FPR)")
#     plt.ylabel("True Positive Rate (TPR)")
#     plt.title("Curva ROC")
#     plt.legend(loc="lower right")
#     plt.grid()
#     plt.show()

#     # Curva de Precisão-Revocação
#     precision, recall, _ = precision_recall_curve(Y_test, y_pred_proba,pos_label='1')
#     pr_auc = auc(recall, precision)
#     plt.figure(figsize=(8, 6))
#     plt.plot(recall, precision, label="AUC PR = {:.2f}".format(pr_auc))
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title("Curva de Precisão-Revocação")
#     plt.legend(loc="upper right")
#     plt.grid()
#     plt.show()
# else:
#     print("O conjunto de teste não contém ambas as classes, portanto a curva ROC e a Curva de Precisão-Revocação não podem ser exibidas.")

# # Análise por Subgrupos: Exemplo de taxa de evasão por faixa etária
# # age_evasion = dataframeCopia.groupby("idade")["evaded"].mean()
# # plt.figure(figsize=(10, 6))
# # sns.barplot(x=age_evasion.index, y=age_evasion.values)
# # plt.title("Taxa de Evasão por Faixa Etária")
# # plt.xlabel("Faixa Etária")
# # plt.ylabel("Taxa de Evasão")
# # plt.show()

# # Análise de Confusão com Limiar Customizado
# # threshold = 0.3  # Exemplo de limiar customizado
# # Y_pred_custom = (y_pred_proba >= threshold).astype(int)
# # cnf_matrix_custom = metrics.confusion_matrix(Y_test, Y_pred_custom)

# # sns.heatmap(pd.DataFrame(cnf_matrix_custom), annot=True, cmap="YlGnBu", fmt='g')
# # plt.title(f"Matriz de Confusão com Limiar {threshold}")
# # plt.xlabel("Previsão")
# # plt.ylabel("Valor Real")
# # plt.show()

# # Verificação de Overfitting
# train_accuracy = logreg.score(X_train, Y_train)
# test_accuracy = logreg.score(X_test, Y_test)
# print(f"\nAcurácia no Treinamento: {train_accuracy:.2f}")
# print(f"Acurácia no Teste: {test_accuracy:.2f}")

# # Comparação para possíveis sinais de overfitting
# if train_accuracy - test_accuracy > 0.1:
#     print("Possível overfitting: a acurácia no treino é significativamente maior que no teste.")
# else:
#     print("Modelo geral bem ajustado entre treino e teste.")

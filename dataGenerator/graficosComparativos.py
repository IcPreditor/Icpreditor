## Importar módulos necessários
import sys

from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(1, r'processing')
import processing

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_curve, auc

motivo_de_evasao_mapping = {
    "REGULAR": "000000",
    "GRADUADO": "000001",
    "TRANSFERIDO PARA OUTRA IES": "000010",
    "FALECIMENTO": "000011",
    "CANCELAMENTO NOVO CURRICULO": "000100",
    "CANCELAMENTO POR ABANDONO": "000101",
    "CANCELAMENTO DE MATRICULA ": "000110",
    "CANCELAMENTO P/ MUDANCA CURSO": "000111",
    "CANCELAMENTO P/ DECISAO JUDICIAL": "001000",
    "CANCELAMENTO P/ SOLICITACAO ALUNO": "001001",
    "SUSPENSAO TEMPORARIA": "001010",
    "CONCLUIDO - NAO COLOU GRAU": "001011",
    "CANCELAMENTO_NAO_CUMPRIMENTO_PEC": "001100",
    "CANCELADO NOVO INGRESSO MESMO CURSO": "001101",
    "CANCELADO NOVO INGRESSO OUTRO CURSO": "001110",
    "CUMPRIMENTO_CONVENIO": "001111",
    "NAO COMPARECEU CADASTRO": "010000",
    "REMANEJADO_CURSO_OU_PERIODO": "010001",
    "NAO_COMPARECEU_AO_REMANEJAMENTO": "010010",
    "INGRESSANTE NAO FEZ 1ª MATRICULA": "010011",
    "TERMINO_DO_INTERCAMBIO": "010100",
    "GRADUADO - DECISAO JUDICIAL": "010101",
    "CANCELADO REPROVOU TODAS POR FALTAS": "010110",
    "SUSPENSO_DEBITO_BIBLIOTECA": "010111",
    "CANCELADO 3 REPROV MESMA DISCIPLINA": "011000",
    "SOLICITOU DESVÍNCULO SEM MATRÍCULA": "011001",
    "AGUARDANDO_CADASTRAMENTO": "011010",
    "VALIDACAO_DECLARACAO_PENDENTE": "011011"
}
motivo_de_evasao_mapping = dict(zip(motivo_de_evasao_mapping.values(),motivo_de_evasao_mapping.keys()))
print(motivo_de_evasao_mapping)
# Carregar dados processados e fazer a divisão treino/teste
X_train, Y_train, X_test, Y_test, dataframeCopia, feature_cols = processing.getInputOutput(undersampling=False, regressao=True)

# Converter Y_train e Y_test para inteiros para garantir a consistência
# Y_train = Y_train.astype(int)
# Y_test = Y_test.astype(int)

# Verificar e converter `X_train` e `X_test` para numérico
# X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
# X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

X_test_motivo = X_test['motivo_de_evasao']
X_test = X_test.drop(columns=['motivo_de_evasao'])
X_train = X_train.drop(columns=['motivo_de_evasao'])
Y_test = Y_test.drop(columns=['motivo_de_evasao'])
Y_train = Y_train.drop(columns=['motivo_de_evasao'])

Y_test.reset_index(inplace=True,drop=True)
Y_train.reset_index(inplace=True,drop=True)

# Usar One-Hot Encoding para lidar com variáveis categóricas
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Ignorar categorias desconhecidas
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)


X_test_motivo = pd.DataFrame(data=X_test_motivo)
X_test_motivo.reset_index(inplace=True,drop=True)
X_test_motivo['resultado'] = 'NaN'
print(X_test_motivo)

# Treinar o modelo de Regressão Logística
logreg = LogisticRegression(random_state=16, max_iter=100000)
logreg.fit(X_train_encoded, Y_train)


# Fazer previsões no conjunto de teste
Y_pred = logreg.predict(X_test_encoded)


# 0 / positivo / Não Evasão
# 1 / negativo / Evasão
#X_train_motivo['resultado']='NaN'
labels = ['True (Dropout)','False (Non-dropout)','Falso Positivo','Falso Negativo']

X_test_motivo['motivo_de_evasao'] = X_test_motivo['motivo_de_evasao'].map(motivo_de_evasao_mapping)

for i in range(len(Y_pred)):
    if Y_pred[i] == Y_test.iloc[i]:

        if Y_pred[i]=='0':
            X_test_motivo.at[i,'resultado'] = labels[1] # Negativo | False (Non-dropout)
        else:
            X_test_motivo.at[i,'resultado'] = labels[0] # Positivo | True (Dropou)
    else:
        if Y_pred[i]=='0':
            X_test_motivo.at[i,'resultado'] = labels[3] # Falso Negativo
        else:
            X_test_motivo.at[i,'resultado'] = labels[2] # Falso Positivo
print(X_test_motivo.value_counts('motivo_de_evasao'))

# 1. Gráfico de Barras - Contagem de Falsos Negativos
fn_count = X_test_motivo[X_test_motivo['resultado'] == labels[3]]['motivo_de_evasao'].value_counts()

nums = list(range(1,len(fn_count)+1))

print(fn_count.values)
# Criando o gráfico de barras
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=nums, y=fn_count.values, color='lightblue',)

ax.set_title('Contagem de Falsos Negativo por Motivo de Evasão')
ax.set_xlabel('Motivo de Evasão')
ax.set_ylabel('Contagem de Falsos Negativo')
# Adicionando uma legenda manualmente

cat_leg = [f'{num}: {cat}' for num,cat in zip(nums,fn_count.index)]
handles = [mpatches.Patch(color='lightblue',label=label) for label in cat_leg]
ax.legend(handles=handles, title='Motivos de Evasão', loc='upper right')

plt.show()

# 3. Heatmap - Frequência de Falsos Negativos
cores_personalizadas = LinearSegmentedColormap.from_list('branco_azul', ['#EEEEEE', '#0000EE'])
#X_test_motivo = X_test_motivo[~X_test_motivo["resultado"].isin([labels[0],labels[1],labels[3]])]
heatmap_data = pd.crosstab(X_test_motivo['motivo_de_evasao'],X_test_motivo['resultado'])

plt.figure(figsize=(16, 9))
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap=cores_personalizadas)
plt.title('Frequência de Resultados por Categoria')
plt.xlabel('Resultado')
plt.ylabel('Motivos de Evasão')
plt.show()
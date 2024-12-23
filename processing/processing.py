#Data processing, mapping and transformation on the bases we use
#Eduardo Augusto, 2024

import json
import pandas as pd
import numpy as np
# Mapeamento de dados do bando


#    # A coluna 'evaded' contem 1 para evadidos e 0 para não evadidos
#    evadidos = dataframe[dataframe['evaded'] == 1]
#    nao_evadidos = dataframe[dataframe['evaded'] == 0]
#   LINHA 117 # Realizando um undersampling, que seria diminuir da classe maior para igualar com a menor
#    if len(evadidos) < len(nao_evadidos):
#        nao_evadidos_balanced = nao_evadidos.sample(n=len(evadidos), random_state=42)
#        dataframe_balanced = pd.concat([evadidos, nao_evadidos_balanced])
#    else:
#        evadidos_balanced = evadidos.sample(n=len(nao_evadidos), random_state=42)
#        dataframe_balanced = pd.concat([evadidos_balanced, nao_evadidos])
#
#    # Embaralhar o dataframe
#    dataframe_balanced = dataframe_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
#
# Carrega dicionario com codeCurso/turno


def load_large_json(file_path):
    return pd.read_json(file_path)


# Mapeamento Setor
setor_mappint = { # Atualizado Igual
    "CH - CENTRO DE HUMANIDADES": "001",
    "CCT - CENTRO DE CIÊNCIAS E TECNOLOGIA":"010",
    "CEEI - CENTRO DE ENGENHARIA ELÉTRICA E INFORMÁTICA":"011",
    "CTRN - CENTRO DE TECNOLOGIA E RECURSOS NATURAIS":"100",
    "CCBS - CENTRO DE CIÊNCIAS BIOLÓGICAS E DA SAÚDE":"101",
    "CENTRO PARFOR":"110"
}

# Mapeamento binário para valores categóricos
binary_mappings = {
    "genero": {"MASCULINO": "0", "FEMININO": "1"}, #Atualizado Igual
    "estado_civil": {"SOLTEIRO": "001", "CASADO": "010",  "VIÚVO": "100", "DIVORCIADO": "011","SEPARADO JUDICIALMENTE":"011", "-": "000"}, #Atualizado
    "situacao": {"INATIVO": "0", "ATIVO": "1"},
    "politica_afirmativa": {"BON. ESTADUAL": "00010", "L1": "00011", "L2": "00100", "L5": "00101", "L6": "00110", "L9": "00111", "L10": "01000", "L13": "01001", "L14": "01010", "-": "00001", "LI_PPI": "01011", "LI_PCD": "01100", "LI_EP": "01101", "LB_EP": "01110", "LB_Q": "01111", "LB_PPI": "10000", "LB_PCD": "10001"},
    "tipo_de_ensino_medio": {"SOMENTE ESCOLA PRIVADA": "010", "SOMENTE ESCOLA PÚBLICA": "001", "PÚBLICA E PRIVADA, TENDO FICADO MAIS TEMPO EM ESCOLA PÚBLICA": "011", "PÚBLICA E PRIVADA, TENDO FICADO MAIS TEMPO EM ESCOLA PRIVADA": "100", "-": "000"}, # Atualizado
    "cor": {"BRANCA": "001", "PRETA": "010", "PARDA": "100", "AMARELA": "101", "NÃO DECLARADA": "000", "INDÍGENA":"011","-": "000"},# Atualizado
    "prac_deficiente": {"NÃO": "01", "SIM": "10", "-": "00"},# Atualizado
    "turno_do_curso":{"INTEGRAL":'11', "MATUTINO":'01', "NOTURNO":"10"} #Atualizado Igual
}

#Mapeamento binário somente para motivo de evasão, pois é muito extenso
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

#Ajuste do tipo_de_ensino_medio desconhecido
def adjust_secondary_school_type(row):
    if row["tipo_de_ensino_medio"] == "DESCONHECIDA":
        if row["politica_afirmativa"] == "A0":
            return "PRIVADA"    
        else:
            return "PUBLICA"
    return row["tipo_de_ensino_medio"]
# Classificação de taxa de sucesso
def taxa_binary(taxa_sucesso): #Atualizado Igual
    if taxa_sucesso <= 0.2:
        return '001'
    elif taxa_sucesso <=0.4:
        return '010'
    elif taxa_sucesso <=0.6:
        return '011'
    elif taxa_sucesso <=0.8:
        return '100'
    elif taxa_sucesso <=1.0:
        return '101'
# classificação de cra
def cra_binary(cra): #Atualizado Igual
    if cra <= 5:
        return '001'
    elif cra <=7:
        return '010'
    elif cra <=8:
        return '011'
    elif cra <=10:
        return '100'
#Classificação de faixa etária
def age_to_binary(idade): #Atualizado Igual
    if idade <= 15:
        return '001'
    elif idade <= 18 :
        return '010'
    elif idade <= 21:
        return '011'
    elif idade <= 25:
        return '100'
    elif idade <= 30:
        return '101'
    elif idade <= 40:
        return '110'
    else:
        return '111'
    
#Classificação de renda per capita
def income_to_binary(prac_renda_per_capita_ate):
    if prac_renda_per_capita_ate <= 1.5:
        return '001'
    elif prac_renda_per_capita_ate <= 3.0 :
        return '010'
    elif prac_renda_per_capita_ate <= 5.0:
        return '011'
    elif prac_renda_per_capita_ate <= 7.0:
        return '100'
    elif prac_renda_per_capita_ate <= 10.0:
        return '101'
    else:
        return '110'

def mapeamentoDataframe(dataframe):
     #Fazer lista de turnos
    dataframe["tipo_de_ensino_medio"] = dataframe.apply(adjust_secondary_school_type, axis=1)

    #Transformando todos os inputs em maiúsculo para padronizar o mapeamento
    dataframe = dataframe.map(lambda x: x.upper() if isinstance(x, str) else x)

    #Transformando todas as idades em inteiro antes de realizar o mapeamento
    dataframe['idade'] = dataframe['idade'].astype(int)
    #Transformando todas as taxas de sucesso em float antes de realizar mapeamento
    dataframe['taxa_de_sucesso'] = dataframe['taxa_de_sucesso'].astype(float)
    #Transformando todas as craditos_do_cra em inteiro antes de realizar mapeamento
    dataframe['creditos_do_cra'] = dataframe['creditos_do_cra'].astype(int)
    #Transformando todas as taxas de sucesso em float antes de realizar mapeamento
    dataframe['notas_acumuladas'] = dataframe['notas_acumuladas'].astype(float)


    # Converter valores categóricos para sua representação binária
    for column, mapping in binary_mappings.items():
        dataframe[column] = dataframe[column].map(mapping)

    # Aplicar o mapeamento binário para a coluna 'motivo_de_evasao'
    dataframe['motivo_de_evasao'] = dataframe['motivo_de_evasao'].map(motivo_de_evasao_mapping)
    # Aplicar o mapeamento binário para a coluna 'nome_do_setor'
    dataframe['nome_do_setor'] = dataframe['nome_do_setor'].map(setor_mappint)
    # Converter valores numéricos para binário
    # idade
    dataframe["idade"] = dataframe["idade"].apply(age_to_binary)
    # taxa de sucesso
    dataframe['taxa_de_sucesso'] = dataframe['taxa_de_sucesso'].apply(taxa_binary)
    # renda
    dataframe["prac_renda_per_capita_ate"] = dataframe["prac_renda_per_capita_ate"].apply(income_to_binary)

    # calcular cra com notas_acumuladas e creditos_do_cra
    dataframe['cra'] = (dataframe["notas_acumuladas"]/dataframe["creditos_do_cra"]).apply(cra_binary)
    return dataframe

def getInputOutput(undersampling=True,regressao=False,evadedColumn=False):
    # Caminho para o arquivo JSON
    file_path = r'data/students.json'
    
    #Carregar os dados
    dataframe = load_large_json(file_path)
    # Aleatoriza banco de dados
    dataframe = dataframe.sample(frac=1,ignore_index=True,random_state=100)
    
    dataframe = mapeamentoDataframe(dataframe)
    # 
    # Definindo as categorias a serem removidas em Motivo de Evasão
    categorias_para_remover = [
        "000100",
        "001010",
        "001101",
        "010000",
        "010011",
        "011010"
    ]

    # Removendo os registros do DataFrame
    dataframe = dataframe[~dataframe['motivo_de_evasao'].isin(categorias_para_remover)]

    # Fazer uma lista dos estudantes que evadiram com base na sua razão de inatividade
    dataframe['evaded'] = dataframe.apply(lambda row: 0 if row['motivo_de_evasao'] in ['000001', '010101'] else 1, axis=1)

    # Verificar a proporção de evadidos
    print(dataframe.value_counts("evaded"))

    # Colunas a serem mantidas
    columns_to_keep = ["idade", "genero", "estado_civil", "politica_afirmativa", "tipo_de_ensino_medio", "turno_do_curso", "cor", "prac_renda_per_capita_ate", "prac_deficiente", "nome_do_setor",'taxa_de_sucesso','cra','evaded','motivo_de_evasao']
    if(evadedColumn):
        columns_to_keep.append('evaded')
        columns_to_keep.append('nome_do_curso')
    dataframeCopia = dataframe
    # Remover todas as colunas que não estão em columns_to_keep
    # Dataframe se torna dataframe_balanced
    dataframe = dataframe.drop(columns=[column for column in dataframe.columns if column not in columns_to_keep])
    dataframe = dataframe.astype(str)
    # Substituir NaN por binário de 0s
    dataframe.fillna("0", inplace=True)

    

    #Aplicar Undersampling se requisitado
    if(undersampling):
        contagem = dataframe.value_counts("evaded")
        if contagem["0"]>contagem["1"]:
            classMin = "1"
            classMax = "0"
        else:
            classMin = "0"
            classMax = "1"
        minimo = contagem[classMin]
        dfMin = dataframe[dataframe["evaded"]==classMin]
        dfMax = dataframe[dataframe["evaded"]==classMax].sample(n=minimo,random_state=0)
    
        #Dividir Teste Treino
        dfTreinoMin = dfMin.sample(frac=0.7, random_state=0)  # 70% para treino
        dfTesteMin = dfMin.drop(dfTreinoMin.index)  # O restante para teste (30%)

        dfTreinoMax = dfMax.sample(frac=0.7, random_state=0)  # 70% para treino
        dfTesteMax = dfMax.drop(dfTreinoMax.index)  # O restante para teste (30%)

        dfTreino = pd.concat([dfTreinoMin,dfTreinoMax])
        dfTeste = pd.concat([dfTesteMin,dfTesteMax])

        dfTreino = dfTreino.sample(frac=1,random_state=0,ignore_index=True)
        dfTeste = dfTeste.sample(frac=1,random_state=0,ignore_index=True)
    else:
        # Não aplicar undersampling
        dfTreino = dataframe.sample(frac=0.7, random_state=0)  # 70% para treino
        dfTeste = dataframe.drop(dfTreino.index)
        
    

    
    Y_treino = dfTreino['evaded']
    X_treino = dfTreino.drop(columns=['evaded'])

    Y_teste = dfTeste['evaded']
    X_teste = dfTeste.drop(columns=['evaded'])
    
    if(not regressao):
        Y_treino = Y_treino.to_list()
        X_treino = X_treino.apply(lambda row: "".join(row.values), axis=1).to_list()

        Y_teste = Y_teste.to_list()
        X_teste = X_teste.apply(lambda row: "".join(row.values), axis=1).to_list()

        # Transformando objetos em lista dos caractestes de si proprio (Necessário Artmap)
        Y_treino = [list(aux1) for aux1 in Y_treino]
        X_treino = [list(aux2) for aux2 in X_treino]

        Y_teste = [list(aux1) for aux1 in Y_teste]
        X_teste = [list(aux2) for aux2 in X_teste]
    columns_to_keep.remove('evaded')
    return(X_treino, Y_treino,X_teste,Y_teste,dataframeCopia,columns_to_keep)
getInputOutput(undersampling=True,regressao=False)
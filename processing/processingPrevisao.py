#Data processing, mapping and transformation on the bases we use
#Eduardo Augusto, 2024

import json
import pandas as pd
import numpy as np
from datetime import datetime as dt
# Mapeamento de dados do bando
# idade faixas <=15:001, <=18:010, <=21:011, <=25:100, <=30:101, <=40:110, >40:111
# genero masculino:00 feminino:01 outro:10 descinhecido:11
# estado_civil solteiro:000 casado:001 separado:010 viuvo:011 divorciado:100 desconhecido:101
# politica_afirmativa "A0": "0000", "L1": "0001", "L2": "0010", "L5": "0011", "L6": "0100", "L9": "0101", "L10": "0110", "L13": "0111", "L14": "1000", "BONUS": "1001"
# tipo_de_ensino_medio "PRIVADA": "000", "PUBLICA": "001", "MAJORITARIAMENTE_PUBLICA": "010", "MAJORITARIAMENTE_PRIVADA": "011", "DESCONHECIDA": "100"
# turno: "Integral":'11',"Matutino":'01',"Noturno":"10"
# nacionalidade: brasileira:0 estrangeira:1
# cor: "BRANCA": "001", "PRETA": "010", "PARDA": "10", "INDÍGENA": "011", "MESTIÇA": "100", "ORIENTAL": "101", "OUTRAS": "110"
# prac_renda_per_capita_ate: Faixa de renda, semelhante ao idade
# prac_deficiencias: "-": "0", "Sim": "1"

#    # A coluna 'evaded' contem 1 para evadidos e 0 para não evadidos
#    evadidos = dataframe[dataframe['evaded'] == 1]
#    nao_evadidos = dataframe[dataframe['evaded'] == 0]

def load_large_json(file_path):
    return pd.read_json(file_path)


# Mapeamento Setor
setor_mappint = {
    "CH - CENTRO DE HUMANIDADES": "001",
    "CCT - CENTRO DE CIÊNCIAS E TECNOLOGIA":"010",
    "CEEI - CENTRO DE ENGENHARIA ELÉTRICA E INFORMÁTICA":"011",
    "CTRN - CENTRO DE TECNOLOGIA E RECURSOS NATURAIS":"100",
    "CCBS - CENTRO DE CIÊNCIAS BIOLÓGICAS E DA SAÚDE":"101",
    "CENTRO PARFOR":"110"
}

# Mapeamento binário para valores categóricos
binary_mappings = {
    "genero": {"MASCULINO": "0", "FEMININO": "1"},
    "nacionalidade": {"BRASILEIRA": "0", "ESTRANGEIRA": "1"},
    "estado_civil": {"SOLTEIRO": "000", "CASADO": "001",  "VIÚVO": "010", "DIVORCIADO": "011","SEPARADO JUDICIALMENTE":"011", "-": "101"},
    "situacao": {"INATIVO": "0", "ATIVO": "1"},
    "politica_afirmativa": {"BON. ESTADUAL": "00000", "L1": "00001", "L2": "00010", "L5": "00011", "L6": "00100", "L9": "00101", "L10": "00110", "L13": "00111", "L14": "01000", "-": "01001", "LI_PPI": "01010", "LI_PCD": "01011", "LI_EP": "01100", "LB_EP": "01101", "LB_Q": "01110", "LB_PPI": "01111", "LB_PCD": "10001"},
    "tipo_de_ensino_medio": {"SOMENTE ESCOLA PRIVADA": "000", "SOMENTE ESCOLA PÚBLICA": "001", "PÚBLICA E PRIVADA, TENDO FICADO MAIS TEMPO EM ESCOLA PÚBLICA": "010", "PÚBLICA E PRIVADA, TENDO FICADO MAIS TEMPO EM ESCOLA PRIVADA": "011", "-": "100"},
    "cor": {"BRANCA": "000", "PRETA": "001", "PARDA": "010", "AMARELA": "011", "NÃO DECLARADA": "100", "INDÍGENA":"101","-": "111"},
    "prac_deficiente": {"NÃO": "00", "SIM": "01", "-": "10"},
    "turno_do_curso":{"INTEGRAL":'11', "MATUTINO":'01', "NOTURNO":"10"}
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
def taxa_binary(taxa_sucesso):
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
def cra_binary(cra):
    if cra <= 5:
        return '001'
    elif cra <=7:
        return '010'
    elif cra <=7:
        return '011'
    elif cra <=8:
        return '100'
    elif cra <=10:
        return '101'
    else:
        return '000'
#Classificação de faixa etária
def age_to_binary(idade):
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
#Calcular Idade baseado na Data de Nascimento
def calcular_idade(data_nascimento):
    data_nascimento = dt.strptime(data_nascimento, "%Y-%m-%d %H:%M:%S.%f")
    hoje = dt.now()
    idade = hoje.year - data_nascimento.year
    if (hoje.month, hoje.day) < (data_nascimento.month, data_nascimento.day):
        idade -= 1
    return idade

def getInputOutput():
    # Caminho para o arquivo JSON
    file_path = r'data/studentsPred.json'
    
    #Carregar os dados
    dataframe = load_large_json(file_path)
    #Fazer lista de turnos
    dataframe["tipo_de_ensino_medio"] = dataframe.apply(adjust_secondary_school_type, axis=1)

    #Transformando todos os inputs em maiúsculo para padronizar o mapeamento
    dataframe = dataframe.map(lambda x: x.upper() if isinstance(x, str) else x)

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
    # Gera coluna idade basa
    
    dataframe['idade'] = dataframe['data_de_nascimento'].apply(calcular_idade)
    # Converter valores numéricos para binário
    # idade
    
    dataframe["idade"] = dataframe["idade"].apply(age_to_binary)
    # taxa de sucesso
    dataframe['taxa_de_sucesso'] = dataframe['taxa_de_sucesso'].apply(taxa_binary)
    # renda
    dataframe["prac_renda_per_capita_ate"] = dataframe["prac_renda_per_capita_ate"].apply(income_to_binary)

    # calcular cra com notas_acumuladas e creditos_do_cra
    dataframe['cra'] = (dataframe["notas_acumuladas"]/dataframe["creditos_do_cra"]).apply(cra_binary)
    
    # Colunas a serem mantidas
    columns_to_keep = ['matricula_do_estudante',"idade", "genero", "estado_civil", "politica_afirmativa", "tipo_de_ensino_medio", "turno_do_curso", "cor", "prac_renda_per_capita_ate", "prac_deficiente", "nome_do_setor",'taxa_de_sucesso','cra']
    # Remover todas as colunas que não estão em columns_to_keep
    # Dataframe se torna dataframe_balanced
    dataframeCopia = dataframe
    dataframe = dataframe.drop(columns=[column for column in dataframe.columns if column not in columns_to_keep])
    dataframe = dataframe.astype(str)
    # Substituir NaN por binário de 0s
    dataframe.fillna("0", inplace=True)
    
    dataframe = dataframe.reindex(['matricula_do_estudante','turno_do_curso','nome_do_setor','estado_civil','genero','idade','cor','tipo_de_ensino_medio','politica_afirmativa','prac_renda_per_capita_ate','taxa_de_sucesso','prac_deficiente','cra'], axis=1)
    X_previsao = dataframe
    return(X_previsao,dataframeCopia,columns_to_keep)
getInputOutput()
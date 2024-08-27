#Data processing, mapping and transformation on the bases we use
#Eduardo Augusto, 2024

import json
import pandas as pd
import numpy as np
from dicTurno import getTurno
# Mapeamento de dados do bando
# age faixas <=15:001, <=18:010, <=21:011, <=25:100, <=30:101, <=40:110, >40:111
# gender masculino:00 feminino:01 outro:10 descinhecido:11
# maritalStatus solteiro:000 casado:001 separado:010 viuvo:011 divorciado:100 desconhecido:101
# affirmativePolicy "A0": "0000", "L1": "0001", "L2": "0010", "L5": "0011", "L6": "0100", "L9": "0101", "L10": "0110", "L13": "0111", "L14": "1000", "BONUS": "1001"
# secondarySchooltype "PRIVADA": "000", "PUBLICA": "001", "MAJORITARIAMENTE_PUBLICA": "010", "MAJORITARIAMENTE_PRIVADA": "011", "DESCONHECIDA": "100"
# turno:     "M":'00', "V":'01', "N":'10', "D":'11' obs:extraido do curso 
# nationality: brasileira:0 estrangeira:1

# Carrega dicionario com codeCurso/turno
dictionaryTurnos = getTurno()

def whatTurno(row):
    return dictionaryTurnos[str(row["courseCode"])]

def load_large_json(file_path):
    return pd.read_json(file_path)


# Mapeamento binário para valores categóricos
binary_mappings = {
    "gender": {"MASCULINO": "00", "FEMININO": "01", "OUTRO": "10", "DESCONHECIDO": "11"},
    "nationality": {"BRASILEIRA": "0", "ESTRANGEIRA": "1"},
    "maritalStatus": {"SOLTEIRO": "000", "CASADO": "001", "SEPARADO": "010", "VIUVO": "011", "DIVORCIADO": "100", "DESCONHECIDO": "101"},
    "status": {"GRADUADO": "00", "ATIVO": "01", "INATIVO": "10"},
    "inactivityReason": {"ABANDONO": "000", "DESCONHECIDO": "001", "TRANSFERENCIA": "010", "CONCLUIU_MAS_NAO_COLOU_GRAU": "011", "DESISTENCIA": "100"},
    "affirmativePolicy": {"A0": "0000", "L1": "0001", "L2": "0010", "L5": "0011", "L6": "0100", "L9": "0101", "L10": "0110", "L13": "0111", "L14": "1000", "BONUS": "1001"},
    "secondarySchoolType": {"PRIVADA": "000", "PUBLICA": "001", "MAJORITARIAMENTE_PUBLICA": "010", "MAJORITARIAMENTE_PRIVADA": "011", "DESCONHECIDA": "100"}
}

#Ajuste do secondarySchoolType desconhecido
def adjust_secondary_school_type(row):
    if row["secondarySchoolType"] == "DESCONHECIDA":
        if row["affirmativePolicy"] == "A0":
            return "PRIVADA"    
        else:
            return "PUBLICA"
    return row["secondarySchoolType"]

#Classificação de faixa etária
def age_to_binary(age):
    if age <= 15:
        return '001'
    elif age <= 18 :
        return '010'
    elif age <= 21:
        return '011'
    elif age <= 25:
        return '100'
    elif age <= 30:
        return '101'
    elif age <= 40:
        return '110'
    else:
        return '111'

def getInputOutput():
    # Caminho para o arquivo JSON
    file_path = r'data/students.json'
    
    #Carregar os dados
    dataframe = load_large_json(file_path)
    # Aleatoriza banco de dados
    dataframe = dataframe.sample(frac=1,ignore_index=True,random_state=100)
    #Fazer lista de turnos
    dataframe["turno"] = dataframe.apply(lambda row: whatTurno(row) , axis=1)

    dataframe["secondarySchoolType"] = dataframe.apply(adjust_secondary_school_type, axis=1)

    # Converter valores categóricos para sua representação binária
    for column, mapping in binary_mappings.items():
        dataframe[column] = dataframe[column].map(mapping)

    # Converter valores numéricos para binário
    dataframe["age"] = dataframe["age"].apply(age_to_binary)

    #Fazer uma lista dos estudantes que evadiram com base na sua razao de inatividade
    dataframe['evaded'] = dataframe.apply(lambda row: '1' if row['status'] == '10' and row['inactivityReason'] != '010' else '0', axis=1)
    evaded_list = dataframe['evaded'].tolist()
    print(dataframe.value_counts("evaded"))
    # Colunas a serem mantidas
    columns_to_keep = ["age", "gender", "nationality", "maritalStatus", "affirmativePolicy", "secondarySchoolType", "turno"]

    # Remover todas as colunas que não estão em columns_to_keep
    dataframe = dataframe.drop(columns=[column for column in dataframe.columns if column not in columns_to_keep])

    # Substituir NaN por binário de 0s
    dataframe.fillna("0", inplace=True)
    dataframe = dataframe.astype(str)
    #Concatenar os resultados em uma string binária por linha
    binary_strings = dataframe.apply(lambda row: "".join(row.values), axis=1).to_list()
    #print(dataframe)
    return(binary_strings,evaded_list)

getInputOutput()
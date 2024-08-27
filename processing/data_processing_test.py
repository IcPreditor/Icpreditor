#Fake data just to test the processing.py
#Eduardo Augusto, 2024

import pandas as pd

def get_student_data():
    data = [
        {
            "age": 35,
            "gender": "FEMININO",
            "nationality": "BRASILEIRA",
            "maritalStatus": "DESCONHECIDO",
            "status": "INATIVO",
            "inactivityReason": "DESISTENCIA",
            "affirmativePolicy": "BONUS",
            "secondarySchoolType": "DESCONHECIDA",
            "courseCode": "13301100",
            "curriculumCode": "2017"
        },
        {
            "age": 22,
            "gender": "MASCULINO",
            "nationality": "BRASILEIRA",
            "maritalStatus": "SOLTEIRO",
            "status": "ATIVO",
            "inactivityReason": None,
            "affirmativePolicy": "L1",
            "secondarySchoolType": "PUBLICA",
            "courseCode": "13311110",
            "curriculumCode": "2018"
        },
        {
            "age": 27,
            "gender": "OUTRO",
            "nationality": "BRASILEIRA",
            "maritalStatus": "CASADO",
            "status": "INATIVO",
            "inactivityReason": "TRANSFERENCIA",
            "affirmativePolicy": "A0",
            "secondarySchoolType": "PRIVADA",
            "courseCode": "13301300",
            "curriculumCode": "2019"
        },
        {
            "age": 30,
            "gender": "DESCONHECIDO",
            "nationality": "ESTRANGEIRA",
            "maritalStatus": "DIVORCIADO",
            "status": "GRADUADO",
            "inactivityReason": None,
            "affirmativePolicy": "L6",
            "secondarySchoolType": "MAJORITARIAMENTE_PUBLICA",
            "courseCode": "13301400",
            "curriculumCode": "2020"
        },
        {
            "age": 29,
            "gender": "MASCULINO",
            "nationality": "BRASILEIRA",
            "maritalStatus": "SEPARADO",
            "status": "INATIVO",
            "inactivityReason": "ABANDONO",
            "affirmativePolicy": "L9",
            "secondarySchoolType": "MAJORITARIAMENTE_PRIVADA",
            "courseCode": "13301500",
            "curriculumCode": "2021"
        }
    ]
    return pd.DataFrame(data)
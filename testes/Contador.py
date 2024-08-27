#Count occurrences of certain parameters
#Guilherme Fernandes, 2024
'''
import json

estadoCivil = 0
nascimento = 0
escola = 0
genero = 0
affirmativePolicy = 0

total = 0

for aluno in json.load(open("data/students.json")):
    total += 1
    if aluno["maritalStatus"]=="DESCONHECIDO":
        estadoCivil += 1
    if aluno["placeOfBirth"] == None:
        nascimento += 1
    if aluno["secondarySchoolType"] == "DESCONHECIDA":
        escola += 1
    if aluno["gender"] != "FEMININO" and aluno["gender"] != "MASCULINO":
        genero += 1
          
print("Contagem Desconhecidos: ")
print("Estado Civil: "+str(estadoCivil)+"/"+str(total))
print("Tipo de Ensino: "+str(escola)+"/"+str(total))
print("nascimento: "+str(nascimento)+"/"+str(total))
print("nascimento: "+str(genero)+"/"+str(total))
'''
import numpy as np
from python_artmap import ARTMAPFUZZY
import bancoFalso as bf
import sys

sys.path.insert(1, r'C:\Users\m5253\VisualStudio\Icpreditor\processing')
import processing
#All Students Data
students_in,students_out = processing.getInputOutput()
#students_in = [list(item[0]) for item in students_in]
#len_students = len(students_in)
for i in range(len(students_in)):
    if(len(students_in[i])!=19):
        print("{0}##{1}".format(i,len(students_in[i])))
##Training Data
input = np.array(students_in)
#input = input.astype(int)
output = np.array(students_out)

#print(students_in)

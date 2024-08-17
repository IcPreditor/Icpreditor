#Test values of rhoARTa and rhoARTb
import numpy as np
from python_artmap import ARTMAPFUZZY
import bancoFalso as bf
import sys

sys.path.insert(1, r'C:\Users\m5253\VisualStudio\Icpreditor\processing')
import processing
#All Students Data
students_in,students_out = processing.getInputOutput()
students_in = [list(item[0]) for item in students_in]
len_students = len(students_in)

##Training Data
input = np.array(students_in[0:len_students//2])
input = input.astype(int)
output = np.array(students_out[0:len_students//2])

##Testing Data
teste_in = np.array(students_in[len_students//2:len_students-1])
teste_in = teste_in.astype(int)
teste_out = np.array(students_out[len_students//2:len_students-1])

if output[0][0]==0:
    aux3 = [0,1]
else:
    aux3 = [1,0]

#rhoARTa AND rhoARTb
ra = 0.1
rb = 0.1

print("##Iniciando Treinamento##")
print("RhoA: " + str(ra))
print("RhoB: " + str(rb))
ArtMap = ARTMAPFUZZY(input, output, rhoARTa=ra, rhoARTb=rb)
ArtMap.train()
print("##Treinamento Finalizado##")

#Number of elements in the array of test
num_ele = len(teste_in)

sucesso = 0
amostra_nao_evasao = 0
sucesso_nao_evasao = 0
amostra_evasao = 0
sucesso_evasao = 0

print("##Iniciando Teste##")

for y in range(num_ele):
    resultado = ArtMap.test(teste_in[y]).get("index")
    if(teste_out[y]==0):
        amostra_nao_evasao += 1 
    elif(teste_out[y]==1):
        amostra_evasao += 1

    if (teste_out[y]==aux3[resultado]):
        sucesso += 1

        if(teste_out[y]==0):
            sucesso_nao_evasao += 1 
        elif(teste_out[y]==1):
            sucesso_evasao += 1

print("##Teste Finalizado##")
erro_evasao = amostra_evasao-sucesso_evasao
erro_nao_evasao = amostra_nao_evasao-sucesso_nao_evasao

print("Resultados:")
print("{0:#^8} | {1:^19} | {2:^19} |".format("","Evasão","Não Evasão"))
print("{0:8} | {1:^8} | {2:^8} | {3:^8} | {4:^8} |".format("","qtd","%","qtd","%"))
print("{0:^8} | {1:^8} | {2:^8} | {3:^8} | {4:^8} |"
      .format("Amostra",amostra_evasao,100.0,amostra_nao_evasao,100.0))
print("{0:^8} | {1:^8} | {2:^8} | {3:^8} | {4:^8} |".
      format("Acerto",sucesso_evasao,(sucesso_evasao/amostra_evasao)*100,sucesso_nao_evasao,(sucesso_nao_evasao/amostra_nao_evasao)*100))
print("{0:^8} | {1:^8} | {2:^8} | {3:^8} | {4:^8} |"
      .format("Erro",erro_evasao,(erro_evasao/amostra_evasao)*100,erro_nao_evasao,(erro_nao_evasao/amostra_nao_evasao)*100))
print(sucesso/num_ele)
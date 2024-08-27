# Auxiliary functions for testing
#Guilherme Fernandes, 2024
import numpy as np
from random import choices,randint
from python_artmap import ARTMAPFUZZY

#Generates a output (1 character binary string) based on input (n character binary string)
def genOutput(input):
    #print(a)
    if len(input)==1:
        return input[0]
    else:
        aux = []
        aux2 = 0
        for i in range(0,len(input),2):
            if input[i]==input[i+1]:
                aux.append(0)
            else: 
                aux.append(1)
            aux2+=1
        return genOutput(aux)
            
#Generates a database for testing
def bancoFalso(num,size_word=16):
    input_aux = []
    output_aux = []
    for x in range(num):
        a = choices(range(2),k=size_word)
        b = genOutput(a)
        input_aux.append(a)
        output_aux.append([b])

    input = np.array(input_aux)
    output = np.array(output_aux)
    
    return (input,output)

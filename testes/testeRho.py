#Test values of rhoARTa and rhoARTb
#Eduardo Augusto, 2024
#Guilherme Fernandes, 2024
import numpy as np
from python_artmap import ARTMAPFUZZY
import bancoFalso as bf
import sys
import tabelaGenerator
#Testing values of RhoARTa and RhoARTb between 0.1 and 0.999
tabelaGenerator.treinamento_teste(0.1,0.1)
for rb_int in range(10,1,-2):
    rb = rb_int/10.0
    for ra_int in range(10,1,-2):
        ra = ra_int/10.0
        #Training
        tabelaGenerator.treinamento_teste(ra,rb)
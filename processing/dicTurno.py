#Functions over courses
#Guilherme Fernandes, 2024
import json
binaryTurnos = {
    "M":'00',
    "V":'01',
    "N":'10',
    "D":'11'
}

dicTurno = {}

def getTurno():
    for course in json.load(open("data/coursesActives.json", encoding='utf-8')):
        # Exceções Medicina Engenharia Elétrica Matématica
        if course['code']=="12205100" or course['code']=="21205100" or course['code']=="19107110" or course['code']=="14123100": #exceções
            dicTurno[course['code']] = '11'
        else:
            turno = binaryTurnos[course["name"][-1]]
            dicTurno[course['code']] = turno
    return dicTurno
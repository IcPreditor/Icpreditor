#python3
#Generates .json with all students
#Uses requests to the eureca ufcg system, 
#with the credentials in credentials.json, the token is saved in token.json
#Guilherme Fernandes, 2024

import requests as req
import pandas as pd
import json


url_token = 'https://pre.ufcg.edu.br:8443/as_scao'
url_eureca = 'https://eureca.sti.ufcg.edu.br/das'

#request token
def genToken(data):
    request = req.post((url_token+"/as/tokens"), json=data )
    #token = r.json()['token']
    #saves token in token.json
    with open("data/token.json","w") as token_file:
        json.dump(request.json(),token_file)
    #retorna token
    return request.json()["token"]

#request profile (using token)
def getProfile(token):
    headers = {'content-type':'application/json',"authentication-token":token}
    request = req.get((url_token+"/as/profile"),headers=headers)
    return request.json()

#request Courses Actives (using token)
def saveCoursesActives(token):
    headers = {'content-type':'application/json',"authentication-token":token}
    request = req.get(url_eureca+"/courses/getActives",headers=headers)
    #save coursesActives
    with open("data/coursesActives.json","w") as courses_file:
            json.dump(request.json(),courses_file)
    #return r.content

#student by course // 2017.1-2023.2
def saveStudents(token):
    headers = {'content-type':'application/json',"authentication-token":token}
    students = []
    for course in json.load(open("data/coursesActives.json")):
        request = req.get(url_eureca+"/students?courseCode="+course["code"]+"&from=2023.1&to=2023.2&anonymize=true",headers=headers)
        students.extend(request.json()["students"])
    with open("data/students.json","w") as students_file:
        json.dump(students,students_file)

#For test

#Opens file with credentials (credentials.json)
credentials = open("data/credentials.json","r")

#loads .json
data = json.load(credentials)

#request token 
token = genToken(data)

#close file dados.json
credentials.close()

#get profile
print(getProfile(token))
#get courses
print(saveCoursesActives(token))
#get students 2017
print(saveStudents(token))
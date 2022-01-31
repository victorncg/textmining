# FRAUD DETECTION

# Este código foi criado com o objetivo de identificar padrões em textos de redes sociais para 
# sinalizar possíveis fraudes. Os padrões que estão sendo procurados aqui foram antes estudados
# para entender a relação que possuiam com fraudes de fato

# Versão 1.2.0

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
path = os.getcwd()

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np
from bs4 import BeautifulSoup
import ast
import sys
import bleach
import time
from PIL import Image
import string
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from bleach.sanitizer import Cleaner
cleaner = Cleaner()
from unidecode import unidecode
import pyodbc
import sqlalchemy as sa
from dateutil.parser import parse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit


execfile('funcoes.py')



bad_chars = ['@font-face', 'span.EstiloDeEmail', '<html>','<head>','&nbsp;','<b>','</b>','<a href="mailto:>','</a>', '<a href="','<a>','<a','a>',
            '><b>','</a><b></b>','<i>','&lt;','</a>&gt;;','</strong','<strong'
            'mailto:','&gt;','&#43;','>','</i>','</i','"','&amp;','&nbsp','\n']






inicio = 20191104
fim = 20191131
x = 20000
cols = ['email', 'rfc_pred', 'Atenção', 'Cadastral', 'Comunicação', 'Interno AAI', 'Movimentações', 'Notificações', 'Notícias', 'Spam', 'Transferência', 'Vazio', 'Verificação', 'Data',
 'From', 'To', 'MessageId', 'E-mail original']

basefull = pd.DataFrame(columns=cols)


# ====================================================================================
# EXECUÇÃO DO CÓDIGO
import warnings

inittime = time.time()

server = ''
database = ''
username = ''
password = ''
driver= '{ODBC Driver 13 for SQL Server}'

strconn = 'DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password
cnn = pyodbc.connect(strconn)

for i in range(inicio,fim):
    reftime = time.time()
    print("Início de execução do código para a base",i)
    string = 'SELECT *  FROM [].[dbo].['
    query = string + str(i) + ']'
    #print(query)
    df = pd.read_sql_query(query, cnn)
    print("Checando se o dataframe está vazio")
    if not df.empty:
        print("Base",i,"aberta")
        warnings.filterwarnings("ignore")
        classificadas = exec_inteiro(df,x)
        print("Base",i,"teve as seguintes dimensões: ",classificadas.shape)
        basefull =  basefull.append(classificadas)
        print("Tempo de execução do código para a base",i,"foi de",round((time.time() - reftime),2),"segundos")
    if df.empty:
        print("DATAFRAME ESTÁ VAZIO")
    print("============================================================================================")
    print(" ")
        
print("Tempo de execução do código inteiro foi de",round((time.time() - inittime),2),"segundos")

baseact = basefull



# ==================================================================================================
#  SALVANDO A BASE RESULTANTE COMO OUTPUT

for col in ['email','rfc_pred','E-mail original']:
    basefull[col] = basefull[col].apply(unidecode)
    
anomes = str(inicio)[0:6]
string_save = "emails_classificados_" + anomes + "_2701_" +".csv"
basefull.to_csv(string_save, sep = ';',encoding = 'utf-8',decimal = ',')


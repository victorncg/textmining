# TEXT MINING
# Código para classificação de mensagens
# Esse código é utilizado para classificar e-mails novos com um modelo já treinado previamente
# Ele foi criado em 2019 com o objetivo de auxiliar na detecção de anomalias através
# de mensagens trocadas
# ===================================================================================================

# Número de e-mails
a = 100000

# Truncagem do tamanho dos e-mails
x = 20000

# 1. ABERTURA DE BIBLIOTECAS
# Abrindo biblioteca que faz paralelização automática

print("============================================================================================")

print(" ")

print("1. ABRINDO BIBLIOTECAS")

import time
start_time = time.time()

import pandas as pd
import multiprocessing
import pickle
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
from PIL import Image
import string
import matplotlib.pyplot as plt
from IPython import get_ipython
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
import matplotlib.pyplot as plt
import seaborn as sns

import os
path = os.getcwd()


# Exibição do resultado do código
print("1. Tempo de abertura das bibliotecas foi de ","%s segundos" % (time.time() - start_time))
print(" ")


# ==================================================================================================================
# 2. ESTABELECENDO CONEXÕES COM AS BASES DE DADOS

print("2. ESTABELECENDO CONEXÕES COM AS BASES DE DADOS")

print("2. Vamos abrir ", str(a), " mensagens")

basestime = time.time()

server = ''
database = ''
username = ''
password = ''
driver= '{}'
strconn = 'DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password
cnn = pyodbc.connect(strconn)

querya = "SELECT TOP ("
queryb = ") *  FROM [].[dbo].[]"

query = querya + str(a) + queryb

df = pd.read_sql_query(query, cnn)

print("2. ", str(a), " Mensagens abertas")

# Exibição do resultado do código
print("2. Tempo de estabelecimento das conexões com as bases de dados foi de ","%s segundos" % (time.time() - basestime))

print(" ")

# ==================================================================================================================
# 3. REMOÇÃO DE VARIÁVEIS
# Variáveis mantidas para serem processadas posteriormente mas que serão removidas:

print("3. REMOÇÃO DE VARIÁVEIS")

remocaotime = time.time()

df = df.drop(['LastModifiedDateTime', 'ReceivedDateTime',
              'SentDateTime', 'InternetMessageId',
              'ParentFolderId', 'IsDeliveryReceiptRequested',
              'ConversationIndex', 'MentionsPreview',
              'FromName', 'FromEmailAddress',
              'ToEmailAddress', 'ToName'], axis = 1)

df = df.drop(['WebLink', 'InferenceClassification',
              'BodyPreview', 'BodyContentType',
              'ChangeKey', 'ConversationId',
              'FlagStatus', 'HasAttachments',
              'Id'], axis = 1)

print("3. Tempo de remoção das variáveis foi de ","%s segundos" % (time.time() - remocaotime))

print(" ")

# ==================================================================================================================
# 4. INÍCIO DO PROCESSAMENTO DOS E-MAILS
# Variáveis mantidas para serem processadas posteriormente mas que serão removidas:

print("4. INÍCIO DO PROCESSAMENTO DOS E-MAILS")

inicioproctime = time.time()

# 3. Início do Processamento dos E-mails
# 3.1. Obtenção de métricas básicas
# 3.1.1. Tamanho do data frame
# 3.1.2. Comprimento médio dos e-mails

df["comp_assunto"]= df.Subject.str.len() 
df["comp_email"]= df.BodyContent.str.len() 
df["comp_email"].mean()

# 3.2. Criação de uma função que consolida o pré-processamento
# 3.2.1. Definir elementos externos para filtro de texto
# Uma vez que finalizamos o estudo da estrutura dos e-mails, precisamos agora padronizar os métodos de limpeza para repetí-los para todos os e-mails. Nessa etapa, replicaremos todas as etapas de limpeza dos e-mails 
# para facilitar o entendimento:

#    a. Captura do e-mail
#    b. Remoção do excesso de HTML
#    c. Remoção de caracteres HTML que não foram removidos na etapa "b"
#    d. Remoção de strings de configuração de e-mail
#    e. Remoção de pontuação que poderia causar problemas na exportação

# a. bad_chars

# initializing bad_chars_list 
bad_chars = ['@font-face', 'span.EstiloDeEmail', '<html>','<head>','']

# initializing string_email list 
string_email = ['&nbsp;','<b>','</b>','<a href="mailto:>','</a>', '<a href="','<a>','<a','a>',
            '><b>','</a><b></b>','<i>','&lt;','</a>&gt;;','</strong','<strong'
            'mailto:','&gt;','&#43;','>','</i>','</i','"','&amp;']

def limpeza(n):
    
    c = bleach.clean(n, strip=True)

    for i in bad_chars : 
        c = c.replace(i, '') 
    
    for i in string_email : 
        c = c.replace(i, ' ')     

    d = c.replace(';', ',')
    d = d.replace('\r', '')
    d = d.replace('\n', '')
    
    return d

print("4. Processamento inicial de e-mails levou ","%s segundos" % (time.time() - inicioproctime))

print(" ")

# ==================================================================================================================
# 5. LIMPEZA PRELIMINAR DE TEXTO
# Variáveis mantidas para serem processadas posteriormente mas que serão removidas:

print("5. LIMPEZA PRELIMINAR DE TEXTO")

limpezaprelimtime = time.time()

print("5. Adicionando etapa de corte de texto e truncando em ", x, " caracteres")

df['BodyContent'] = df['BodyContent'].str[:x]

print("5. Truncagem finalizada. Dando início ao processo de limpeza.")

df['limpo'] = 0

for label,row in df.iterrows():
    df.loc[label,'limpo'] = limpeza(row['BodyContent'])
    
df["tamanho_limpo1"]= df.limpo.str.len() 
df["reducao1"]= df["tamanho_limpo1"]/df["comp_email"]

df = df.drop(['Subject', 'BodyContent'], axis = 1) 

# 3.5. Remoção de acentos
# Essa etapa serve para que o texto fique padronizado e facilite o mapeamento de padrões posteriormente

df['limpo'] = df['limpo'].str.lower()

print("5. Limpeza preliminar dos e-mails tomou ","%s segundos" % (time.time() - limpezaprelimtime))

print(" ")

# ==================================================================================================================
# 6. CRIAÇÃO DE FUNÇÕES DE APOIO
# Variáveis mantidas para serem processadas posteriormente mas que serão removidas:

print("6. CRIAÇÃO DE FUNÇÕES DE APOIO")

funcoesapoiotime = time.time()

def find_return(str1):
    
    # Remoção de disclaimers e trechos padrão
    str2 = re.sub('Agente[^>]+Autonomo[^>]+atua[^>]+CVM[^>]+3710', '', str1).strip()
    str2 = re.sub('http.{0,60}click[^>]+CONTIDAS[^>]+COMPLEMENTARES[^>]+3710[^>]+VAREJO', '', str2).strip()
    str2 = re.sub('Esse[^>]+email[^>]+enviado[^>]+para[^>]+@}', '', str2).strip()
    
    str2 = re.sub('clique[^>]+aqui[^>]+recuperar[^>]+senha[^>]+automatic', '', str2).strip()
    str2 = re.sub('aspxf', '', str2).strip()
    
    str2 = re.sub('http:[^>]+xpi[^>]+SP[^>]+75[^>]+Miami[^>]+10017', '', str2).strip()
    str2 = re.sub('http:[^>]+facebook[^>]+instagram[^>]+anexos[^>]+lei[^>]+dirigida[^>]+contidas[^>]+autor[^>]+XP[^>]+3710', '', str2).strip()
    str2 = re.sub('Important[^>]+any[^>]+may[^>]+only[^>]+accept[^>]+copy[^>]+written[^>]+XP[^>]+lost[^>]+e-mail', '', str2).strip()
    str2 = re.sub('Obter[^>]+https[^>]+Outlook[^>]+Android', '', str2).strip()
    str2 = re.sub('Este.{0,10}material[^>]+XP[^>]+nao[^>]+deve[^>]+ser[^>]+indicati[^>]+garantia[^>]+fundo[^>]+responsabiliza[^>]+3710[^>]+FUNDO[^>]+FGC[^>]+VAREJO[^>]+garantia[^>]+receber[^>]+e-mails', '', str2).strip()
    
    str2 = re.sub('saiba.{0,3}mais[^>]+disclaimer[^>]+xp[^>]+auxiliar[^>]+risco[^>]+signatario[^>]+cumprimento[^>]+apimec[^>]+ancord[^>]+suitability[^>]+futuros[^>]+provaveis[^>]+termo[^>]+commodity[^>]+consubstanciar[^>]+desempenho.{0,5}investimento', '', str2).strip()
    str2 = re.sub('disclaimer[^>]+xp[^>]+auxiliar[^>]+risco[^>]+signatario[^>]+cumprimento[^>]+apimec[^>]+ancord[^>]+suitability[^>]+futuros[^>]+provaveis[^>]+termo[^>]+commodity[^>]+consubstanciar[^>]+desempenho.{0,5}investimento', '', str2).strip()
    str2 = re.sub('esta instituicao.{0,5}aderente.{0,5}codigo anbima.{0,5}regulacao.{0,5}melhores praticas.{0,7}atividade.{0,5}distribuicao.{0,5}produtos.{0,5}investimento.{0,5}varejo', '', str2).strip()
    
    # Remoção de códigos de clientes
    str2 = re.sub('cliente \d{6}', '', str2).strip()
    str2 = re.sub('Cliente \d{6}', '', str2).strip()
    str2 = re.sub('codigo \d{7}', '', str2).strip()
    str2 = re.sub('codigo \d{6}', '', str2).strip()
    str2 = re.sub('conta.{0,6}\d{6}.{0,3}', '', str2).strip()
    
    # Remoção de horas    
    str2 = re.sub('\d{2}:\d{2}', '', str2).strip()
    str2 = re.sub('\d{2}h.{0,1}', '', str2).strip()
 
    # Remoção de caracteres especiais 
    str2 = re.sub('CPA.{0,1}\d{2}', '', str2).strip()
    str2 = re.sub('&#\d{5}', '', str2).strip()
    str2 = re.sub('image\d{3}.jpg', '', str2).strip()
    str2 = re.sub('image\d{3}.png', '', str2).strip()
    str2 = re.sub('Instrucao Normativa.{0,10} \d{3}/\d{2}', '', str2).strip()
    
    # Remoção de telefones
    str2 = re.sub('\d{2}.{0,2}\d{4}.{0,1}\d{4}', '', str2).strip()
    str2 = re.sub('\([^)]*\).{0,3}\d{5}.{0,3}\d{4}', '', str2).strip()
    str2 = re.sub('\([^)]*\).{0,3}\d{4}.{0,3}\d{4}', '', str2).strip()
    str2 = re.sub('\d{2}\d{5}-\d{4}', '', str2).strip()   
    str2 = re.sub('\d{2}\d{4}-\d{4}', '', str2).strip()  
    str2 = re.sub('\d{4}.{0,4}\d{5}', '', str2).strip()
    str2 = re.sub('\d{4}.{0,3}\d{4}', '', str2).strip()
    str2 = re.sub('\d{4}.{0,3}\d{3}.{0,3}\d{4}', '', str2).strip()    
    str2 = re.sub('VOIP.{0,3}\d{4}.{0,3}', '', str2).strip()
    
    # Remoção de datas
    str2 = re.sub('\d{2}/\d{2}/\d{4}', '', str2).strip()
    #str2 = re.sub('\d{2}/\d{1}/\d{4}', '', str2).strip()
    #str2 = re.sub('\d{1}/\d{2}/\d{4}', '', str2).strip()
    #str2 = re.sub('\d{1}/\d{1}/\d{4}', '', str2).strip()
    str2 = re.sub('\d{2}/\d{2}/\d{2}', '', str2).strip()    
    str2 = re.sub('\d{2}/\d{2}/\d{4}', '', str2).strip()    
    str2 = re.sub('\d{2}-\d{2}-\d{4}', '', str2).strip()    
    str2 = re.sub('Em \d{1} de.{0,5}de \d{4}', '', str2).strip()    
    str2 = re.sub('Em.{0,5} \d{2} de.{0,5}de \d{4}', '', str2).strip()    
    str2 = re.sub('em \d{1} de.{0,5}de \d{4}', '', str2).strip()    
    str2 = re.sub('em.{0,5} \d{2} de.{0,5}de \d{4}', '', str2).strip()    
    str2 = re.sub('milhoes', '000000', str2).strip()
    str2 = re.sub('mm ', '000000', str2).strip()
    
    # Remoção de CEPS
    str2 = re.sub('\d{5}-\d{3}', '', str2).strip()
    
    str3 = str2.replace('.', '')
    
    # Regular Expression that matches digits in between a string 
    return (str3)

# Definição da função "find_sum"
def find_sum(str1):
        
    # Regular Expression that matches digits in between a string 
    return sum(map(int,re.findall('\d+',str1)))

print("6. Criação das funções de apoio levou ","%s segundos" % (time.time() - funcoesapoiotime))

print(" ")
# ==================================================================================================================
# 7. APLICAÇÃO DA FUNÇÃO FIND_RETURN
# A aplicação dessa função é uma das etapas mais importantes desse código pois ela é responsável por remover grande parte dos trechos de texto que são desnecessários

print("7. APLICAÇÃO DA FUNÇÃO FIND_RETURN")

aplicafuntime = time.time()

df['limpo'] = df['limpo'].astype(str)

df['email_limpo'] = df['limpo'].apply(find_return)

print("7. Tempo de processamento da função find_return foi de ","%s segundos" % (time.time() - aplicafuntime))

print(" ")

# ==================================================================================================================
# 8. VARIÁVEIS DE CARACTERIZAÇÃO DE TAMANHO
# Variáveis mantidas para serem processadas posteriormente mas que serão removidas:
     
print("8. VARIÁVEIS DE CARACTERIZAÇÃO DE TAMANHO")

vartamtime = time.time()

df["tamanho_limpo2"]= df.email_limpo.str.len() 

df["reducao2"]= (df["tamanho_limpo2"]+1)/(df["comp_email"]+1)

df["reducao_relativa"]= (df["tamanho_limpo2"]+1)/(df["tamanho_limpo1"]+1)

# Depois de terminar todo o processamento, vamos mudar a ordem das variáveis na base para garantir que as três primeiras variáveis da base sejam sempre:
# MessageId
# CreatedDateTime
# email_limpo

df = df[['MessageId', 
 'CreatedDateTime',
 'email_limpo',
 'Importance',
 'IsReadReceiptRequested',
 'IsRead',
 'IsDraft',
 'UnsubscribeEnabled',
 'comp_assunto',
 'comp_email',
 'limpo',
 'tamanho_limpo1',
 'reducao1',
 'tamanho_limpo2',
 'reducao2',
 'reducao_relativa'
 ]]

df = df.drop(['limpo'], axis = 1)

print("8. Tempo de criação das variáveis de caracterização de tamanho foi de ","%s segundos" % (time.time() - vartamtime))

print(" ")

# ==================================================================================================================
# 9. CRIAÇÃO DE VARIÁVEIS DE CARACTERIZAÇÃO DE TEXTO
# Variáveis mantidas para serem processadas posteriormente mas que serão removidas:

print("9. CRIAÇÃO DE VARIÁVEIS DE CARACTERIZAÇÃO DE TEXTO")

varcaractexttime = time.time()

df['SUM_VALUE'] = df['email_limpo'].apply(find_sum)
df['POSI_EXC']  = df['email_limpo'].str.find('!')
df['POSI_ARR']  = df['email_limpo'].str.find('@')
df['POSI_HASH'] = df['email_limpo'].str.find('#')
df['POSI_CIF']  = df['email_limpo'].str.find('$')
df['POSI_PER']  = df['email_limpo'].str.find('%')
df['POSI_ECO']  = df['email_limpo'].str.find('&')
df['POSI_PON']  = df['email_limpo'].str.find('.')
df['POSI_PAR1'] = df['email_limpo'].str.find('(')
df['POSI_PAR2'] = df['email_limpo'].str.find(')')
df['POSI_SUM']  = df['email_limpo'].str.find('+')

# 5. Feature Engineering - variáveis de flag
df['FLAG_EXC']  = df['email_limpo'].str.count('!')
df['FLAG_ARR']  = df['email_limpo'].str.count('@')
df['FLAG_HASH'] = df['email_limpo'].str.count('#')
df['FLAG_CIF']  = df['email_limpo'].str.count('$')
df['FLAG_PER']  = df['email_limpo'].str.count('%')
df['FLAG_ECO']  = df['email_limpo'].str.count('&')
df['FLAG_PON']  = df['email_limpo'].str.count('.')
df['ass_disclaimers'] = df['email_limpo'].str.contains('Agente[^>]+Autonomo[^>]+atua[^>]+CVM[^>]+3710', regex=True).astype(int)
df['SOMA'] = df['FLAG_EXC'] + df['FLAG_ARR'] + df['FLAG_HASH'] + df['FLAG_CIF'] + df['FLAG_PER'] + df['FLAG_ECO']

# 6. Feature Engineering - variáveis temáticas
df['prez_aai']      = df['email_limpo'].str.contains('rezad.{0,5} ass', regex=True).astype(int)
df['ATT']           = df['email_limpo'].str.contains('ATT|att |atenciosamente|Atenciosamente', regex=True).astype(int)
df['HTML']          = df['email_limpo'].str.contains('<b|</b|<a|</a|<strong|</strong|strong', regex=True).astype(int)
df['Cump']          = df['email_limpo'].str.contains('bom dia|boa tarde|boa noite|ola|oi', regex=True).astype(int)
df['Trigger']       = df['email_limpo'].str.contains('trigger', regex=True).astype(int)
df['Pessoal']       = df['email_limpo'].str.contains('tudo bem|bom dia|boa tarde|tudo bom|estou bem|como vai', regex=True).astype(int)
df['Agradecimento'] = df['email_limpo'].str.contains('agrade|obrigad', regex=True).astype(int)

print("9. Tempo de criação das variáveis de caracterização de texto foi de ","%s segundos" % (time.time() - varcaractexttime))

print(" ")

# ==================================================================================================================
# 10. INÍCIO DO PROCESSO DE TEXT MINING
# Variáveis mantidas para serem processadas posteriormente mas que serão removidas:

print("10. INÍCIO DO PROCESSO DE TEXT MINING")

iniprotextime = time.time()

train_df = df
punctuation_signs = list("?:!.,;")

print("10. Remoção de sinais")

for punct_sign in punctuation_signs:
    train_df['email_limpo'] = train_df['email_limpo'].str.replace(punct_sign, '')

# Saving the lemmatizer into an object
#wordnet_lemmatizer = WordNetLemmatizer()

nrows = len(train_df)
#lemmatized_text_list = []

#for row in range(0, nrows):
#    
#    # Create an empty list containing lemmatized words
#    lemmatized_list = []
#    
#    # Save the text and its words into an object
#    text = train_df.loc[row]['email_limpo']
#    text_words = text.split(" ")
#
#    # Iterate through every word to lemmatize
#    for word in text_words:
#        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
#        
#    # Join the list
#    lemmatized_text = " ".join(lemmatized_list)
#    
#    # Append to the list containing the texts
#    lemmatized_text_list.append(lemmatized_text)
#    
#    
#train_df['texto_lematizado'] = lemmatized_text_list

# Loading the stop words in portuguese
stop_words = list(stopwords.words('portuguese'))

train_df['email_limpo_filtrado'] = train_df['email_limpo']

print("10. Remoção de stopwords")

for stop_word in stop_words:

    regex_stopword = r"\b" + stop_word + r"\b"
    train_df['email_limpo_filtrado'] = train_df['email_limpo_filtrado'].str.replace(regex_stopword, '')
    
emails_treino = train_df['email_limpo_filtrado']

print("10. Abrindo elemento externo de tfidf")

tfidf_df  = "/home/victor.gomes/tfidf.pickle"

with open(tfidf_df, 'rb') as data:
    tfidf = pickle.load(data)
    
features_test = tfidf.transform(train_df['email_limpo_filtrado']).toarray()

df1 = pd.DataFrame(features_test, columns=tfidf.get_feature_names())

print("10. Unindo base com variáveis criadas previamente e variáveis de tfidf")

res_treino = pd.concat([train_df, df1], axis=1)

id_treino = res_treino['MessageId']

emails_treino = res_treino['email_limpo_filtrado']

print("10. Processo de text mining levou ","%s segundos" % (time.time() - iniprotextime))

print(" ")

# ==================================================================================================================
# 11. INÍCIO DO PAB (Processo de Adequação de Bases)
# Variáveis mantidas para serem processadas posteriormente mas que serão removidas:

print("11. INÍCIO DO PAB")

iniciopab = time.time()

treino_df = "/home/victor.gomes/base_treino.pickle"

with open(treino_df, 'rb') as data:
    treino = pickle.load(data)
    
cols_treino = list(treino.columns.values)

base_alvo = res_treino

for col in cols_treino:
    if col not in base_alvo.columns:
        base_alvo[col] = 0
        
cols_teste = list(base_alvo.columns.values)

for col in cols_teste:
    if col not in treino.columns:
         base_alvo.drop([col], axis = 1, inplace = True)

            
print("11. Processo de adequação de bases levou ","%s segundos" % (time.time() - iniciopab))

print(" ")
            
# ==================================================================================================================            
# 12. INÍCIO DO DEPLOY DO MODELO
# Variáveis mantidas para serem processadas posteriormente mas que serão removidas:

print("12. INÍCIO DO DEPLOY DO MODELO")

inideplmodtime = time.time()

modelo_df = "/home/victor.gomes/random_search.pickle"

with open(modelo_df, 'rb') as data:
    modelo = pickle.load(data)
    
# 12. Início do trecho de medição do tempo de execução
# a. Colunas de probabilidade sendo criadas na base-alvo

print("12. Início da delimitação do valor máximo da variável SUM_VALUE")

basela = base_alvo.fillna(base_alvo.mean())

maxVal = 1000000000000

basela['SUM_VALUE'][basela['SUM_VALUE'] >= maxVal] = maxVal

basela[basela==np.inf]=np.nan

basela.fillna(basela.mean(), inplace=True)

print("12. Início do deploy do modelo")

ML_time = time.time()

teste_result_proba = modelo.predict_proba(basela)

teste_result_proba = pd.DataFrame(teste_result_proba, columns=modelo.classes_)

# b. Variável resposta sendo criada na base-alvo
basela['rfc_pred'] = modelo.predict(basela)

print("12. Carregamento e deploy do modelo levou ","%s segundos" % (time.time() - inideplmodtime))

print(" ")

# ==================================================================================================================     
# Exibição do resultado do código
print("Tempo de execução do código é de ","%s segundos" % (time.time() - start_time), "e o tempo de execução do modelo é de ","%s segundos" % (time.time() - ML_time))

print(" ")

print("============================================================================================")

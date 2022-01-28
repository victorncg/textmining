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





funcoesapoiotime = time.time()

def remove_variaveis(str1):
    
    print("REMOÇÃO DE VARIÁVEIS - Função remove_variaveis()")

    remocaotime = time.time()

    print("Tempo de remoção das variáveis foi de ","%s segundos" % round((time.time() - remocaotime),2))
     
    # Regular Expression that matches digits in between a string 
    return (str2)


def limpa_html(n):

    c = bleach.clean(n, strip=True)

    return c


def trunca_caracters(n,x):
   
    c = n[:x]

    return c

bad_chars = ['@font-face', 'span.EstiloDeEmail', '<html>','<head>','&nbsp;','<b>','</b>','<a href="mailto:>','</a>', '<a href="','<a>','<a','a>',
            '><b>','</a><b></b>','<i>','&lt;','</a>&gt;;','</strong','<strong'
            'mailto:','&gt;','&#43;','>','</i>','</i','"','&amp;','&nbsp','\n']

# c. Criação da função que chamaremos de "limpeza"
def limpeza(n):
    
    for i in bad_chars : 
        c = n.replace(i, ' ') 

    d = c.replace('&nbsp', ' ')
    d = d.replace('nbsp', ' ')
    d = d.replace('&amp', ' ')
    d = d.replace('</b>', ' ')
    d = d.replace('<b>', ' ')
    d = d.replace('\r', ' ')
    d = d.replace('\n', ' ')
    d = d.replace('strong', ' ')
    d = d.replace('<', ' ')
    d = d.replace('>', ' ')
    d = d.replace('&', ' ')
    d = d.replace(';', ' ')
    d = d.replace('a href=', ' ')
    d = d.replace('href=', ' ')
    d = d.replace('href', ' ')
    
    return d

def limpeza_geral(a,x):
    
    b = limpa_html(a)
    
    c = trunca_caracters(b,x)
    
    d = limpeza(c)
    
    return d


def limpa_processa(a,x):
    
    print("INÍCIO DO PROCESSAMENTO DOS E-MAILS")

    inicioproctime = time.time()

    a["comp_assunto"]= a.Subject.str.len() 
    a["comp_email"]= a.BodyContent.str.len() 
    a["comp_email"].mean()

    #print("2. Adicionando etapa de corte de texto e truncando em ", x, " caracteres")
    
    for label,row in a.iterrows():
        a.loc[label,'limpo'] = limpeza_geral(row['BodyContent'],x)

    #print("2. Processamento inicial de e-mails levou ","%s segundos" % (time.time() - inicioproctime))

    #print(" ")

    
    return a

def find_return(str1):
    
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
    str2 = re.sub('\d{5}-\d{3}', ' ', str2).strip()
    
    str3 = str2.replace('.', ' ')
    
    str3 = re.sub(' +', ' ',str3).strip()
    
    # Regular Expression that matches digits in between a string 
    return (str3)

def find_sum(str1):
        
    # Regular Expression that matches digits in between a string 
    return sum(map(int,re.findall('\d+',str1)))

def limpeza_preliminar(n):
   
    #print("5. LIMPEZA PRELIMINAR DE TEXTO")

    limpezaprelimtime = time.time()
    
    n["tamanho_limpo1"]= n.limpo.str.len() 
    n["reducao1"]= n["tamanho_limpo1"]/n["comp_email"]

    ndf = n.drop(['Subject', 'BodyContent'], axis = 1)

    ndf['limpo'] = ndf['limpo'].str.lower()

    #print("3. Limpeza preliminar dos e-mails tomou ","%s segundos" % (time.time() - limpezaprelimtime))

    return ndf

def aplica_findreturn(a):
    
    #print("7. APLICAÇÃO DA FUNÇÃO FIND_RETURN")

    aplicafuntime = time.time()

    a['limpo'] = a['limpo'].astype(str)

    a['email_limpo'] = a['limpo'].apply(find_return)
    
    #print("4. Tempo de processamento da função find_return foi de ","%s segundos" % (time.time() - aplicafuntime))

    #print(" ")
    
    return a

def caracter_tam(a):
    
    #print("8. VARIÁVEIS DE CARACTERIZAÇÃO DE TAMANHO")

    vartamtime = time.time()

    a["tamanho_limpo2"]= a.email_limpo.str.len() 

    a["reducao2"]= (a["tamanho_limpo2"]+1)/(a["comp_email"]+1)

    a["reducao_relativa"]= (a["tamanho_limpo2"]+1)/(a["tamanho_limpo1"]+1)

    # A variável 'limpo' deve ser removida pois não será usada daqui pra frente
    b = a.drop(['limpo'], axis = 1)

    #print("5. Tempo de criação das variáveis de caracterização de tamanho foi de ","%s segundos" % (time.time() - vartamtime))

    #print(" ")
    
    return b


def variav_text(df):
    
    #print("9. CRIAÇÃO DE VARIÁVEIS DE CARACTERIZAÇÃO DE TEXTO")

    varcaractexttime = time.time()

    # Posição de caracteres especiais no texto
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
    df['FLAG_EXC'] = df['email_limpo'].str.count('!')
    df['FLAG_ARR'] = df['email_limpo'].str.count('@')
    df['FLAG_HASH'] = df['email_limpo'].str.count('#')
    df['FLAG_CIF'] = df['email_limpo'].str.count('$')
    df['FLAG_PER'] = df['email_limpo'].str.count('%')
    df['FLAG_ECO'] = df['email_limpo'].str.count('&')
    df['FLAG_PON'] = df['email_limpo'].str.count('.')
    df['SOMA'] = df['FLAG_EXC'] + df['FLAG_ARR'] + df['FLAG_HASH'] + df['FLAG_CIF'] + df['FLAG_PER'] + df['FLAG_ECO']
    df['ass_disclaimers'] = df['email_limpo'].str.contains('Agente[^>]+Autonomo[^>]+atua[^>]+CVM[^>]+3710', regex=True).astype(int)

    # 6. Feature Engineering - variáveis temáticas
    df['prez_aai'] = df['email_limpo'].str.contains('rezad.{0,5} ass', regex=True).astype(int)
    df['ATT'] = df['email_limpo'].str.contains('ATT|att |atenciosamente|Atenciosamente', regex=True).astype(int)
    df['HTML'] = df['email_limpo'].str.contains('<b|</b|<a|</a|<strong|</strong|strong', regex=True).astype(int)
    df['Cump'] = df['email_limpo'].str.contains('bom dia|boa tarde|boa noite|ola|oi', regex=True).astype(int)
    df['Trigger'] = df['email_limpo'].str.contains('trigger', regex=True).astype(int)
    df['Pessoal'] = df['email_limpo'].str.contains('tudo bem|bom dia|boa tarde|tudo bom|estou bem|como vai', regex=True).astype(int)
    df['Agradecimento'] = df['email_limpo'].str.contains('agrade|obrigad', regex=True).astype(int)

    #print("6. Tempo de criação das variáveis de caracterização de texto foi de ","%s segundos" % (time.time() - varcaractexttime))

    #print(" ")
    
    return df


def initextmin(df):
    
    #print("10. INÍCIO DO PROCESSO DE TEXT MINING")
    iniprotextime = time.time()
    train_df = df
    punctuation_signs = list("?:!.,;><")

    #print("10. Remoção de sinais")
    for punct_sign in punctuation_signs:
        train_df['email_limpo'] = train_df['email_limpo'].str.replace(punct_sign, '')

    # Loading the stop words in portuguese
    stop_words = list(stopwords.words('portuguese'))
    train_df['email_limpo_filtrado'] = train_df['email_limpo']

    #print("10. Remoção de stopwords")
    for stop_word in stop_words:

        regex_stopword = r"\b" + stop_word + r"\b"
        train_df['email_limpo_filtrado'] = train_df['email_limpo_filtrado'].str.replace(regex_stopword, '')

    #print("10. Abrindo elemento externo de tfidf")
    tfidf_df  = "/home/victor.gomes/tfidf_20200114.pickle"
    with open(tfidf_df, 'rb') as data:
        tfidf = pickle.load(data)

    features_test = tfidf.transform(train_df['email_limpo_filtrado']).toarray()
    df1 = pd.DataFrame(features_test, columns=tfidf.get_feature_names())

    #print("10. Unindo base com variáveis criadas previamente e variáveis de tfidf")
    res_treino = pd.concat([train_df, df1], axis=1)
    # emails_treino = res_treino['email_limpo_filtrado']

    #print("Processo de text mining levou ","%s segundos" % (time.time() - iniprotextime))
    #print(" ")
    
    #res_treino['emaillimpofinal'] = train_df['email_limpo_filtrado']
    
    return res_treino



def inipro_pab(n):
   
    print("INÍCIO DO PAB")

    iniciopab = time.time()

    treino_df = "/home/victor.gomes/base_treino_20200114.pickle"

    with open(treino_df, 'rb') as data:
        treino = pickle.load(data)

    cols_treino = list(treino.columns.values)
    
    base_alvo = n

    for col in cols_treino:
        if col not in base_alvo.columns:
            base_alvo[col] = 0

    cols_teste = list(base_alvo.columns.values)

    for col in cols_teste:
        if col not in treino.columns:
             base_alvo.drop([col], axis = 1, inplace = True)


    print("Processo de adequação de bases levou ","%s segundos" % (time.time() - iniciopab))

    #print(" ")

    return base_alvo



def deploy_modelo(a,df):
    
    #print("12. INÍCIO DO DEPLOY DO MODELO")

    inideplmodtime = time.time()
    modelo_df = "/home/victor.gomes/random_search_20200114.pickle"

    with open(modelo_df, 'rb') as data:
        modelo = pickle.load(data)

    # 12. Início do trecho de medição do tempo de execução
    # a. Colunas de probabilidade sendo criadas na base-alvo
    
    basealvo_df = "/home/victor.gomes/base_alvo20200124.pickle"

    with open(basealvo_df, 'rb') as data:
        base_alvo = pickle.load(data)

    #print("12. Início da delimitação do valor máximo da variável SUM_VALUE")
    basela = a.fillna(base_alvo.mean())
    maxVal = 1000000000000
    basela['SUM_VALUE'][basela['SUM_VALUE'] >= maxVal] = maxVal
    basela[basela==np.inf]=np.nan
    basela.fillna(basela.mean(), inplace=True)

    #print("12. Início do deploy do modelo")
    ML_time = time.time()
    teste_result_proba = modelo.predict_proba(basela)
    teste_result_proba = pd.DataFrame(teste_result_proba, columns=modelo.classes_)

    # b. Variável resposta sendo criada na base-alvo
    basela['rfc_pred'] = modelo.predict(basela)
    
    base_final = pd.concat([basela, teste_result_proba], axis=1)

    base_final['email'] = df['email_limpo']

    #print("9. Carregamento e deploy do modelo levou ","%s segundos" % (time.time() - inideplmodtime))
    #print(" ")
    
    return base_final




def exec_inteiro(df,x):
    
    reftime = time.time()

    dfa = remove_variaveis(df)
    
    print("Criando coluna limpo")
    dfa['limpo'] = 0

    print("Prosseguindo com a limpeza do e-mail")
    dfe = limpa_processa(dfa,x)

    dfi = limpeza_preliminar(dfe)

    dfo = aplica_findreturn(dfi)

    dfu = caracter_tam(dfo)

    dfax = variav_text(dfu)

    # Dá-se início agora o processo de Text Mining

    tax = initextmin(dfax)
  
    tex = inipro_pab(tax)

    tix = deploy_modelo(tex,dfo)
     
    #print("Especificando variáveis que vão permanecer")
    tox = tix[['email','rfc_pred','Atenção','Cadastral','Comunicação','Interno AAI','Movimentações','Notificações','Notícias','Spam','Transferência','Vazio','Verificação']]
    #print(" ")
    
    #print("Adicionando variável de data")
    tox['Data'] = df['SentDateTime']
    #print(" ")
    
    #print("Adicionando variável de FromEmailAddress")
    tox['From'] = df['FromEmailAddress']
    #print(" ")
    
    #print("Adicionando variável de ToEmailAddress")
    tox['To'] = df['ToEmailAddress']
    #print(" ")
    
    #print("Adicionando variável de MessageId")
    tox['MessageId'] = df['MessageId']
    #print(" ")
    
    #print("Adicionando e-mail original")
    tox['E-mail original'] = dfi['limpo'].str[:x]
    #print(" ")
    
    #print("Adicionando e-mail original")
    #tox['email_limpo_filtrado'] = tax['emaillimpofinal']
    #print(" ")
    
    # Exibição do resultado do código
    #print("Tempo de execução do código foi de ","%s segundos" % (time.time() - reftime))

    return tox


print("Criação das funções de apoio levou ","%s segundos" % (time.time() - funcoesapoiotime))

print(" ")




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


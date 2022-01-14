import requests
from bs4 import BeautifulSoup
import numpy as np
import pytz
import nltk
import spacy
nlp = spacy.load('es_core_news_sm')
def normalize(text): #regresa el texto lematizado como lista
    doc = nlp(text)

    words = [t.orth_ for t in doc if ((not t.is_punct | t.is_stop ) or ( str(t)=='general' or str(t)=='empleo')) ]
    lexical_tokens = [t.lower() for t in words if
                      t.isalpha()]

    return lexical_tokens#lista
#%%

URL = "https://www.eluniversal.com.mx/minuto-x-minuto?seccion=1"
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
page = requests.get(URL,headers=headers)

soup = BeautifulSoup(page.content, "html.parser")
#%%
def scrape_articulo_universal(URL):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    page = requests.get(URL,headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all("div", class_="field field-name-body field-type-text-with-summary field-label-hidden")
    return normalize(results[0].text)
#%%
results = soup.find_all("h2", class_="ce9-Tipo3MinxMin_Titulo")
#%%

titulos=[]
vinculos=[]
contenidos=[]
for x in range(len(results)):
    titulos.append(normalize(results[x].text))
    vinculos.append(results[x].find('a')['href'])
    contenidos.append(scrape_articulo_universal(results[x].find('a')['href']))
    print(x)



#%%


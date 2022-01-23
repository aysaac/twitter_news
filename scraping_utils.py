#%%
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import pytz
# import nltk
import tweepy
import datetime as dt
from datetime import timedelta
import spacy
nlp = spacy.load('es_core_news_sm')
def normalize(text): #regresa el texto lematizado como lista
    doc = nlp(text)

    words = [t.orth_ for t in doc if
             ((not t.is_punct | t.is_stop | (t.pos_=='CCONJ'))
              or ( str(t)=='general' or str(t)=='empleo'))  or t.is_digit]

    lexical_tokens = [t.lower() for t in words if
                     ( t.isalpha() |t.isdigit ())]
    return lexical_tokens#lista


#%%
def scrape_articulo_universal(URL):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    page = requests.get(URL,headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all("div", class_="field field-name-body field-type-text-with-summary field-label-hidden")
    return normalize(results[0].text)
#%%
def google_news_scraping(query,Date):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    # query='Mexico'
    M=str(Date.month)
    D=str(Date.day)
    Y=str(Date.year)
    URL='https://www.google.com.mx/search?q='+query+'&rlz=1C1CHBF_esMX831MX831&biw=1511&bih=730&source=lnt&tbs=cdr%3A1%2Ccd_min%3A'+M+'%2F'+D+ '%2F'+Y+'%2Ccd_max%3A'+M+'%2F'+D+ '%2F'+Y+'&tbm=nws'
    # print(URL)
    page = requests.get(URL, headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")
    res = soup.find_all(role="heading")
    return res

def twitter_results(search_string,until,API):
    # df=pd.DataFrame()
    until=until.strftime('%Y-%m-%d')
    # since=since.strftime('%Y-%m-%d')

    # query='lang:es '+search_string

    return  API.search_tweets(q=search_string,until=until)
#%%
# bearear='AAAAAAAAAAAAAAAAAAAAAKFPXQEAAAAAjCa02471Ahi1pHh5hNKqL9wzvcQ%3Dn0DfJcEyqjbJFxR3Di7ezsTERQOlio1Qhyn4K4IJEDBQ2iu22w'
# acces='1020163577859137537-DedLm3FzPGAYi5uqHhdPh7LegOAGm0'
# secret='Whu92xe1JopsBExmrEKWLcT2YOEKpC2w1M1W9CXkpdKKs'
# auth = tweepy.OAuthHandler('Hxf2c0Mc4ZvRCfG3tgIWDVTRK', 'jj8nBFAiNfINgTsHEgybvWE1ijVFYrPYga0oNs6BPFNA1jkKUM')
#
# # palabras=' OR '.join(str(words) for words in palabras
# # query='lang:es '+' ('+palabras_str+')'
# # query
# API = tweepy.API(auth)
# search_date=dt.date(2021,12,25)
# delta=timedelta(days=1)
# until=search_date+delta
# # since=search_date-delta
# #%%
# h=twitter_results('hey',dt.date.today()-delta,API)
# x=h[-1]
# titles=google_news_scraping('Mexico',search_date)
# normalized_strings=[]
# for x in titles:
#     # print(x)
#     normalized_strings.append(' '.join(normalize(x.text)))
# #%%
# h=twitter_results('contingencia ambiental mexico',dt.date.today()-delta,API)
# h[-1].text
# #%%
# for x in normalized_strings:
#     print(x)
# #%%
# doc=nlp(normalized_strings[-7])
# #%%
# # import matplotlib.pyplot as plt
# from spacy import displacy
# displacy.render(doc, style="ent")
# # plt.show()
# #%%
#   hola mundo

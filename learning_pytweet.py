import tweepy
import datetime as dt
from datetime import timedelta
import pytz
import nltk
import spacy
auth = tweepy.OAuthHandler('Hxf2c0Mc4ZvRCfG3tgIWDVTRK', 'jj8nBFAiNfINgTsHEgybvWE1ijVFYrPYga0oNs6BPFNA1jkKUM')
#%%
nlp = spacy.load('es_core_news_sm')
def normalize(text): #regresa el texto lematizado como lista
    doc = nlp(text)

    words = [t.orth_ for t in doc if ((not t.is_punct | t.is_stop ) or ( str(t)=='general' or str(t)=='empleo')) ]
    lexical_tokens = [t.lower() for t in words if
                      t.isalpha()]

    return lexical_tokens#lista
#%%
# places = api.search_geo(query="MEX", granularity="country")
#
# order=0
# place_id = places[order].id
# print(places[order].full_name)
# print(places[order].bounding_box.corner())
#%%
date=dt.date(2021,12,10)
delta=timedelta(days=5)
#%%
until=date+delta
since=date-delta
palabras=['perros','gatos']
palabras_str=' OR '.join(str(words) for words in palabras)
palabras_str
#%%

from nltk.stem import SnowballStemmer
spanish_stemmer = SnowballStemmer('spanish')

#%%
# texto='Mueren 39 personas en incendio en un barco de Bangladesh'
texto='Por la crisis del sector inmobiliario, los millonarios chinos compran relojes de lujo antes que propiedades'
y=normalize(texto)
query1=[spanish_stemmer.stem(x) for x in y]
query1

#%%
api = tweepy.API(auth)
query='lang:es '+' ('+palabras_str+')'
query
tweets = api.search_tweets(q=query)
#%%
# tweets = api.search_tweets(q=query)


#%%
for tweet in tweets:
    print(tweet.text)
    print(tweet.created_at.astimezone(pytz.timezone('America/Mexico_City')))


import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
nlp = spacy.load('es_core_news_sm')
from scraping_utils import google_news_scraping
#%%
# codiv_patterns = [
#                 {"POS": "PROPN", "pattern": "covid-19"}
#             ]
# ruler = nlp.add_pipe("entity_ruler", after="ner")
# ruler.add_patterns(codiv_patterns)


#%%
matcher=Matcher(nlp.vocab)
pattern1=[
  {"POS": "ADJ", "OP": "+"},
  {"POS": "NOUN", "OP": "+"}]
pattern2=[
  {"POS": "NOUN", "OP": "+"},
  {"POS": "ADJ", "OP": "+"}]
pattern3 = [{'POS': 'VERB',"OP": "+"},{"POS": "NOUN", "OP": "*"}]
# exclduder=[{'LEMMA': {'NOT_IN': ['in']}, "OP": "*"}, {"TEXT": {"REGEX": "^[Mm](\.?|éxico)$"}}]
pattern4 = [  {"POS": "PROPN", "OP": "+"},
            {"POS": "NOUN", "OP": "+"}]

# matcher.add("Propios1", [pattern1])
# matcher.add("Propios2", [pattern2])
# matcher.add("Propios3", [pattern3])
# matcher.add("Propios4", [pattern4])
#%%
doc=nlp('pedro esta comiendo un sandwich')
matches = matcher(doc)
# matches = matcher(doc)
for x in matches:
    print(doc[x[1]:x[2]])
#%%
#%%
import datetime as dt

today=dt.date.today()
news=google_news_scraping('Mexico',today)
#%%
strings=[new.text for new in news]

#%%
for x in strings:
    doc=nlp(x)
    tokens=[dy.orth_ for dy in doc]
    pos=[dy.pos_ for dy in doc]
    print(tokens,end='')
    print('\n ')
    print(pos,end='')
    print('\n __________________________')
#%%
matcher=Matcher(nlp.vocab)
pattern1=[
  {"POS": "ADJ", "OP": "+"},
  {"POS": "NOUN", "OP": "+"}]
pattern2=[
  {"POS": "NOUN", "OP": "+"},
  {"POS": "ADJ", "OP": "+"}]
pattern3 = [{'POS': 'VERB',"OP": "+"},{"POS": "NOUN", "OP": "+"}]
excluder=[{'LEMMA': {'NOT_IN': ['in']}, "OP": "!"}, {"TEXT": {"REGEX": "^[Mm](\.?|éxico)$"}}]
pattern4 = [  {"POS": "PROPN", "OP": "+"},
            {"POS": "NOUN", "OP": "+"}]
pattern5=[{"ENT_TYPE":'LOC'}]
burgir=[{"POS":'NUM'}]
matcher.add('excluir',[excluder])
matcher.add("Propios1", [pattern1])
matcher.add("Propios2", [pattern2])
matcher.add("Propios3", [pattern3])
matcher.add("Propios4", [pattern4])
matcher.add("Propios4", [pattern5])
for x in strings:
    doc=nlp(x)
    tokens=[dy.orth_ for dy in doc]
    matching=matcher(doc)
    pos=[doc[x[1]:x[2]] for x in matching]
    print(tokens,end='')
    print('\n ')
    print(pos,end='')
    print('\n _________________________________________________')

#%%
# nlp(strings[0]).ents[2].label_

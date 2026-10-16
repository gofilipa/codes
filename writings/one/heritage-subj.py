# coding: utf-8
import spacy
nlp = spacy.load("en_core_web_sm")
import pandas as pd
text = pd.read_csv('heritage copy.json')
text = pd.read_json('heritage copy.json')
text.info()
docs = nlp(pipe(text))
docs = nlp(text)
docs = nlp(df['text'])
df = pd.read_csv('heritage copy.json')
df = pd.read_json('heritage copy.json')
text = nlp(df['text'])
df['text']
articles = df['text']
text = nlp(articles)
text = nlp(list(articles))
a = list(articles)
a
a.join(' ')
' '.join(a)
a
len(a)
type(a)
art = ' '.join(a)
type(art)
len(art)
doc = nlp(art)
doc = nlp(art, nlp.max_length = 1800000)
doc = nlp(art, nlp.max_length=1800000)
nlp.max_length = 1800000
doc = nlp(art)
for i in doc[:10]:
    print(i)
for i in doc:
    if i == 'subjective':
        print(i.sent)
for i in doc:
    if 'subjective' in i:
        print(i.sent)
for i in doc:
    if i == 'subjective':
        print(i.sent)
len(i)
len(doc)
for i in doc[:100]:
    print(i)
for i in doc[-100:]:
    print(i)
for i in doc:
    if i.lower_ == 'subjective':
        print(i.sent)
subj_sents = []
for i in doc:
    if i.lower_ == 'subjective':
        subj_sents.append(i.sent)
with open('subjectivity-heritage.org') as f:
    f.writelines(subj_sents)
with open('subjectivity-heritage.org', 'rw') as f:
    f.writelines(subj_sents)
with open('subjectivity-heritage.org', 'w') as f:
    f.writelines(subj_sents)
with open('subjectivity-heritage.org', 'w') as f:
    f.writelines(str(subj_sents))
with open('subjectivity-heritage.org', 'w') as f:
    for i in subj_sents:
        f.write(str(i))
        f.write('/n/n')
with open('subjectivity-heritage.org', 'w') as f:
    for i in subj_sents:
        f.write(str(i))
        f.write('\n\n')

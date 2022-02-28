import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
import nltk
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer

# tokenization

doc1 = "tentative phrase!"
doc2 = "my address is a.p@gmail.com. My website is www.stat.it/home"
doc3 = "the # of students has been constant through the years." 
doc4 = "The initial 'Board of Visitors' included U.S. Presidents Thomas Jefferson, James Madison and James Monroe"
doc5 = "Apple is building its headquarters in China for 100 million dollars"
doc6 = "The student commitee of UAB will meet at Plaza Central at 9pm"
doc7 = ""
doc = nlp(doc4)
for tokens in doc:
    print(tokens.text)

    print(tokens.text, end="|")

print(doc[2]) # doc can be accessed as list
#for ent in doc.ents:
#    print(ent.text)
#    print(ent.label_)
#    print(str(spacy.explain(ent.label_)))

doc = nlp(doc2)

# graph

# displacy.render(doc,style='dep',options={'distance':100})
# displacy.render(doc,style='dep',jupyter=True, options={'distance':100})
# doc.noun_chunks

for token in doc:
    print(token.text)
    print(tokens.pos_)
    print(tokens.dep_)

# for stemming
ps = PorterStemmer()
words = []
ps.stem("run")

ss = SnowballStemmer(language='english')

# stopwords
import spacy
nlp = spacy.load("en_core_web_sm")
print(nlp.Defaults.stop_words)

# is_stop, add, remove
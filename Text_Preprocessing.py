
# coding: utf-8

# In[8]:


import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
import spacy
import json
from urllib.request import urlopen
from unidecode import unidecode
import requests
import html


# In[9]:


#nlp = spacy.load('en')


# In[ ]:


class TextProcessing:
    def __init__(self,appId,data):
        self.appId = appId
        self.data = data

    # segment description into sentences
    def SegmemtintoSentences(self,sents_already_segmented=False):
        self.sents=[]
        self.data = unidecode(self.data)
        pattern=r'".*"(\s+-.*)?'
        self.data = re.sub(pattern, '', self.data)
        pattern1 = r'\'s'
        self.data = re.sub(pattern1,"",self.data)

        if sents_already_segmented == True:       
            list_lines = self.data.split('\n')
            list_lines = [line for line in list_lines if line.strip()!='']
            #self.sents=list_sents
            for line in list_lines:
                if line.strip()=="Credits" or line.strip()=="credits":
                    break
                
                if line.isupper():
                    line = line.capitalize()
                    
                sentences = nltk.sent_tokenize(line)
                self.sents.extend(sentences)
        elif sents_already_segmented==False:
            self.sents = nltk.sent_tokenize(self.data)
            self.sents = [sent for sent in self.sents if sent.strip()!='']
        
    # clean sentences
    def  GetCleanSentences(self):
        sentences=[]
        clean_sentences=[]
        # remove explanations text with-in brackets
        for sent in self.sents:
            sent = html.unescape(sent.strip())
            sent = sent.lstrip('-')
            regex = r"(\(|\[).*?(\)|\])"
            urls = re.findall('(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?',sent)
            emails = re.findall("[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*", sent)         
            match_list = re.finditer(regex,sent)
            
            new_sent=sent
            
            if len(urls)==0 and len(emails)==0 :
                if match_list:
                    for match in match_list:
                        txt_to_be_removed = sent[match.start():match.end()]
                        new_sent=new_sent.replace(txt_to_be_removed,"")
                    
                    clean_sentences.append(new_sent)
                else:
                    clean_sentences.append(sent)
        

        pattern = r'\*|\u2022|#'
        
    
        custom_stop_words = ['anything','beautiful','efficient','enjoyable','way','quick','greeting','features','elegant','instant','fun','price','dropbox','iphone','total','is','in-app','apps','quickly','easily','lovely','others','other','own','the','interesting','addiction','following','featured','best','phone','sense','fantastical','fantastic','better',
                            'award-winning','include','including','winning','improvements','improvement','significant','app','mac','pc','ipad','approach','application','applications','lets','several','safari','pro','google','matter','embarrassing','faster','mistakes','gmail','official','out','results','those','them','have','internet','anymore','are','provide','partial','useful','twitter','facebook','need','lose','it','yahoo','be','swiss','say','makes','make','local','button','will','vary','was','were','cloudapp','everything','straightforward','seamless','mundane','convenience','based','whatever','d','trials','trial','stuff','same','within','paperless','service','use','second','news','easy-to-use','secure','provides','provide','most','common','ask','different','introducing','introduce','ask','no','not','never','allow','accessibility','easy','anyone','subscriptions','losing','ios','function','bar','subscription','requires','require','important']
    
        
        for index,sent in enumerate(clean_sentences):
            clean_sent= re.sub(pattern,"",sent)
            # removing sub-ordinate clauses from a sentence
            sent_wo_clause = self.Remove_SubOrdinateClause(clean_sent)
        
            clean_sentences[index] = sent_wo_clause
            
            tokens = nltk.word_tokenize(clean_sentences[index])
                
            sent_tokens = [w for w in tokens if w.lower() not in custom_stop_words]
            sentences.append(' '.join(sent_tokens))
                
        return sentences
    
    def Remove_SubOrdinateClause(self,sentence):
        sub_ordinate_words= ['when','after','although','because','before','if','rather','since',                            'though','unless','until','whenever','where','whereas','wherever','whether','while','why','which','by','so'
                            ]
        
        sub_ordinate_clause = False
        words=[]
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            if token.lower() in sub_ordinate_words:
                sub_ordinate_clause = True
    
            if sub_ordinate_clause == False:
                    words.append(token)
            elif sub_ordinate_clause == True:
                break
            
        return(' '.join(words).strip())


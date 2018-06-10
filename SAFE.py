
# coding: utf-8

# In[71]:


import spacy
import SAFE_Patterns
from Text_Preprocessing import TextProcessing
from Feature_Matching import Merge_Features
import SAFE_Evaluation
import Feature_Matching
import Text_Preprocessing
import importlib
import json
from urllib.request import urlopen
import re
import requests
import time
import nltk


# In[72]:


importlib.reload(SAFE_Evaluation)
importlib.reload(SAFE_Patterns)
importlib.reload(Text_Preprocessing)
importlib.reload(Feature_Matching)


# In[3]:


nlp = spacy.load('en_core_web_sm')


# In[69]:


class SAFE:
    def __init__(self,appID,data,nlp):
        self.appID=str(appID)
        self.nlp = nlp
        self.data = data
        self.ExtractData()
        
    def ExtractData(self):
        self.app_description = self.data
            
    def PreprocessData(self):
        textProcessor = TextProcessing(self.appID,self.app_description)
        textProcessor.SegmemtintoSentences(sents_already_segmented=True)
        self.clean_sentences = textProcessor.GetCleanSentences()
    
    def ExtractAppFeatures(self):
        SAFE_Patterns_Obj=SAFE_Patterns.SAFE_Patterns(self.appID, self.clean_sentences)
        candidate_features = SAFE_Patterns_Obj.ExtractFeatures_Analyzing_Sent_POSPatterns()
        return candidate_features
    
    def Extract_App_Features(self):
        self.PreprocessData()
        return self.ExtractAppFeatures()


# In[73]:


if __name__ == '__main__':
    data_path = 'app_descriptions_with_manual_feature_extraction.json'
    
    with open(data_path) as json_data:
        data = json.load(json_data)
    
    for app_data in data:
        app_id = app_data['id']
        app_name = app_data['app_name']
        app_true_features = app_data['app_features']
        app_description = app_data['app_description']
        
        
        true_features=[]
        print("#"*5,app_name,"#"*5)

        for t_feature in app_true_features:
            comma_s_pattern = r'\'s'
            true_feature= re.sub(comma_s_pattern,"",t_feature)
            true_features.append(true_feature)

        obj_surf = SAFE(app_id,app_description,nlp)
        extracted_features = obj_surf.Extract_App_Features()

        objEvaluation=SAFE_Evaluation.Evaluate(true_features,extracted_features)
        objEvaluation.PerformEvaluation()

        #print("######################################################")
        print("")


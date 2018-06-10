
# coding: utf-8

# In[1]:


import nltk
from nltk.stem.snowball import SnowballStemmer


# In[ ]:


stemmer = SnowballStemmer("english")


# In[33]:


class Evaluate:
    def __init__(self,true_features,extracted_features):
        self.true_features=true_features
        self.predicted_features = extracted_features
    
    def PerformEvaluation(self):
        tp=0
        fp=0
        fn=0
        
        tp_features_list=[]
        fp_features_list=[]
        fn_features_list=[]
        
        for p_feature in self.predicted_features:
            p_feature_words =  nltk.word_tokenize(p_feature)
            p_feature_words_tag = nltk.pos_tag(p_feature_words)
            p_feature_clean = ' '.join([stemmer.stem(w.lower()) for w,tag in p_feature_words_tag if tag!='PRP$' and tag!='IN'])
        
            found=False
            for t_feature in self.true_features:
                t_feature_words =  nltk.word_tokenize(t_feature)
                t_feature_words_tag = nltk.pos_tag(t_feature_words)
                t_feature_clean = ' '.join([stemmer.stem(w.lower()) for w,tag in t_feature_words_tag if tag!='PRP$' and tag!='IN'])
                
                if p_feature_clean in t_feature_clean:
                    found=True
                    break
            
            if found==True:
                tp = tp + 1
                tp_features_list.append(p_feature)
            
            elif found==False:
                fp = fp + 1
                fp_features_list.append(p_feature)
            
        
        for t_feature in self.true_features:
            t_feature_words =  nltk.word_tokenize(t_feature)
            t_feature_words_tag = nltk.pos_tag(t_feature_words)
            t_feature_clean = ' '.join([stemmer.stem(w.lower()) for w,tag in t_feature_words_tag if tag!='PRP$' and tag!='IN'])
            found=False
            for p_feature in self.predicted_features:
                p_feature_words =  nltk.word_tokenize(p_feature)
                p_feature_words_tag = nltk.pos_tag(p_feature_words)
                p_feature_clean = ' '.join([stemmer.stem(w.lower()) for w,tag in p_feature_words_tag if tag!='PRP$' and tag!='IN'])
        
                if p_feature_clean == t_feature_clean:
                    found=True
                    break
            
            if found==False:
                fn = fn + 1
                fn_features_list.append(t_feature)
                
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
        try:
            fscore = 2*((precision*recall)/(precision+recall))
        except ZeroDivisionError:
            fscore=0.0
        
        print("Precision : %.3f, Recall : %.3f, Fscore: %.3f" % (precision,recall,fscore))


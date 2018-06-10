
# coding: utf-8

# In[1]:

import re
import itertools
import pickle
from nltk.corpus import wordnet as wn
import nltk
import numpy as np
from numpy import linalg as LA
import math


# In[2]:

from enum import Enum
class MERGE_MODE(Enum):
    DESCRIPTION=1
    USER_REVIEWS=2
    DESCRIPTION_USER_REVIEWS=3


# In[3]:

#nlp = spacy.load('en')


# In[15]:

class Merge_Features:
    def __init__(self,appId,mode,nlp):
        self.appId =  appId
        file_path = self.appId.upper() + "_EXTRACTED_APP_FEATURES_"
        
        if mode.value == MERGE_MODE.DESCRIPTION.value:
            self.raw_app_features = self.GetExtractedFeatures(file_path + "DESC.pkl")
        elif mode.value ==  MERGE_MODE.USER_REVIEWS.value:
            self.raw_app_features = self.GetExtractedFeatures(file_path + "REVIEWS.pkl")
        elif mode.value == MERGE_MODE.DESCRIPTION_USER_REVIEWS.value:
            app_features_desc = self.GetExtractedFeatures(file_path + "DESC.pkl")
            app_features_reviews = self.GetExtractedFeatures(file_path + "REVIEWS.pkl")
            
            self.raw_app_features = app_features_desc + app_features_reviews
        
        #print(self.raw_app_features)
        self.nlp = nlp
        
    # matching on a single term level 'send email' and 'email send' are considered matching features
    
    def GetExtractedFeatures(self,path):
        with open (path, 'rb') as fp:
            raw_app_features = pickle.load(fp)
        
        return(raw_app_features)
        
    
    def Merge_Matching_Terms(self):
        
        match_features=[]
        
        for feature1 in self.raw_app_features:
            for feature2 in self.raw_app_features:
                if feature1!=feature2 and (((feature1,feature2) in match_features)==False and ((feature2,feature1) in match_features)==False ) :
                    feature1_words = feature1.split()
                    feature2_words = feature2.split()
                    if len(feature1_words) == len(feature2_words):
                        contain_same_terms = all(w in feature1_words for w in feature2_words)
                    
                        if contain_same_terms==True:
                            match_features.append((feature1,feature2))
                            
        #print("# of features left before term matching are %d" % (len(self.raw_app_features)))
        
        # calcualte frequency and sort list by frequency
        freq_dist = nltk.FreqDist(self.raw_app_features)
        
         #retain only the instance with highest frequency
        for matched_feature in match_features:
            feature1_freq = freq_dist[matched_feature[0]]
            feature2_freq = freq_dist[matched_feature[1]]
            
            if feature1_freq>=feature2_freq:
                del freq_dist[matched_feature[1]]
            else:
                del freq_dist[matched_feature[0]]
        
        # convert to list and sort app features by frequency
        
        app_features_freq = list(freq_dist.items())
        self.app_features_freq = sorted(app_features_freq, key=lambda x: x[1],reverse=True)
        
        #print("# of features left after term matching are %d" % (len(self.app_features_freq)))
      
        #print("")
        print("Set of app features are ->")
        print(self.app_features_freq)
    
    def FeatureExistinSameKey(self,key,new_feature,d):
        found = False
        #print("key->",key)
        #print(d.get(key,'na'))
        #print("#######################")
        if d.get(key,'na') != 'na':
            app_features_list = d[key]
            if app_features_list is not None:
                already_exist = any([app_feature[0]==new_feature for app_feature in app_features_list])
                if already_exist==True:
                    found = True
        else:
            found=True
        
        return(found)
    
    def FeatureExistinDictionary(self, new_feature,d):
        found = False
        for k,v in d.items():
            app_features_list = d[k]
            if app_features_list is not None:
                already_exist = any([app_feature[0]==new_feature for app_feature in app_features_list])
    
                if already_exist==True:
                    found = True
                    break
        
        return(found)
            
    def Merge_Features_based_WordNet(self):
        
        feature_word_synonyms = self.GetFeatureSynonyms()
        
        self.feature_cluster={}
        
        for i in range(0,len(self.app_features_freq)):
            app_feature_1 = self.app_features_freq[i]
            feature1_words = app_feature_1[0].split()
            
            str_feature_1 = ' '.join(feature1_words)
            
            for j in range(0,len(self.app_features_freq)):
                app_feature_2 = self.app_features_freq[j]
                feature2_words = app_feature_2[0].split()
                str_feature_2 = ' '.join(feature2_words)
                found = 0
                if i!=j and len(feature1_words)==len(feature2_words):
                    for k in range(0,len(feature1_words)):
                            syn_features_list=[]
                            feature1_word = feature1_words[k]       
                            feature2_word = feature2_words[k]
                            
                            syn_word_list = set(feature_word_synonyms[feature1_word])
                                   
                            if (feature2_word in syn_word_list)==True:
                                found = found + 1
                               
                    #print(self.raw_app_features[j])
                    #print("found = %d" % (found))
                
                if len(feature1_words)==found and found ==len(feature1_words):
                    if self.feature_cluster.get(str_feature_1,'na') == 'na' and self.FeatureExistinDictionary(str_feature_2,self.feature_cluster)==False:
                        self.feature_cluster[str_feature_1]=[app_feature_1]
                        feature_list = self.feature_cluster[str_feature_1]
                        feature_list.append(app_feature_2)
                        self.feature_cluster[str_feature_1]=feature_list
                    elif self.feature_cluster.get(str_feature_1,'na') != 'na' and self.FeatureExistinDictionary(str_feature_2,self.feature_cluster)==False:
                        feature_list = self.feature_cluster[str_feature_1]
                        feature_list.append(app_feature_2)
                        self.feature_cluster[str_feature_1]=feature_list
            
        # add remaining features
            if self.feature_cluster.get(str_feature_1,'na')=='na' and self.FeatureExistinDictionary(str_feature_1,self.feature_cluster)==False:
                self.feature_cluster[str_feature_1]=[app_feature_1]
        
        #print("group of synonyms app features are ->")
        #print(self.app_features_freq)
        
        #print(self.feature_cluster)
    
    def lemmalist(self,word):
        syn_set = []
        for synset in wn.synsets(word):
            for item in synset.lemma_names():
                syn_set.append(item)
        return syn_set

    def app_feature_words_vector(self,app_feature_words):
        feature_words_vector = [word.vector for word in app_feature_words if word.has_vector]
        if len(feature_words_vector)>0:
            return np.mean(feature_words_vector, axis=0)
        else:
             return(np.array([]))
    
    def GetAppFeatureEmbeddings(self):
        
        app_features_embedding_dict={}
        
        for i in range(0,len(self.app_features_freq)):
            app_feature = (self.app_features_freq[i])[0]
            
            app_feature_vector = self.app_feature_words_vector(self.nlp(app_feature))
            
            if app_feature_vector.size!=0:
                app_features_embedding_dict[app_feature] = app_feature_vector
            else:
                app_features_embedding_dict[app_feature] = np.zeros(300)
            
        return(app_features_embedding_dict)
        

    def cluster_features_word_embeddings(self,similarity_threshold=.70):
        app_features_embedding_dict = self.GetAppFeatureEmbeddings()
        cosine = lambda v1, v2: np.dot(v1, v2) / (LA.norm(v1) * LA.norm(v2))
        
        self.feature_cluster_after_embedding=self.feature_cluster.copy()
        
        #print("")
        #print("group of app features are ->")
        
        for group_key_1,grouped_app_features_1 in self.feature_cluster.items():
            #print(group_key_1, " -> ",grouped_app_features_1)
            embedding_vector_group_key_1 = app_features_embedding_dict[group_key_1]
            
            for group_key_2,grouped_app_features_2 in self.feature_cluster.items():
                if group_key_1!=group_key_2:
                    embedding_vector_group_key_2 = app_features_embedding_dict[group_key_2]
                    dist = cosine(embedding_vector_group_key_1, embedding_vector_group_key_2)
                    
                    if dist>similarity_threshold and self.FeatureExistinSameKey(group_key_1,group_key_2,self.feature_cluster_after_embedding)==False:
                        #print('%s is simialr to %s' % (group_key_1,group_key_2))
                        # merge similar groups 
                        if self.feature_cluster_after_embedding.get(group_key_1,'na')!='na' and self.feature_cluster_after_embedding.get(group_key_2,'na')!='na':
                            app_features_group_1 = self.feature_cluster_after_embedding[group_key_1]                        
                            app_features_group_2 = self.feature_cluster_after_embedding[group_key_2]
                        
                        
                        # group 2 is simialr to group 1 so merge them
                            app_features_group_1.extend(app_features_group_2)
                            del self.feature_cluster_after_embedding[group_key_2]
            
            #print("#####################")
        
    def PrintClusteredFeatures(self,app_features_cluster):
        #print("###############Cluster of app features###########")
        for key_app_feature, cluster_app_features in  app_features_cluster.items():
            print(key_app_feature, " -> ", app_features_cluster[key_app_feature])
            
    def GetFeatureSynonyms(self):
        feature_word_synonyms={}

        for i in range(0,len(self.raw_app_features)):
            #feature_tokens = self.nlp(self.raw_app_features[i])
            feature_tokens = nltk.word_tokenize(self.raw_app_features[i])
            
            for feature_token in feature_tokens:
                
                syn_word_list= set()
                syn_word_list.add(feature_token)
                
                if(feature_word_synonyms.get(feature_token,'na')=='na'):
                    syn_word_list.update(set(self.lemmalist(feature_token)))
                    feature_word_synonyms[feature_token] = syn_word_list

        return(feature_word_synonyms)
    
    def GetAppFeatures_JSONformat(self):
        filepath = self.appId.upper() + "_EXTRACTED_FEATURES.txt" 
        file = open(filepath, 'w')
        
        json_output={}
        json_output['appID'] = self.appId
        
        list_app_features=[]
        features_clusters = []
        for key_app_feature, cluster_app_features in  self.feature_cluster_after_embedding.items():
            feature_dict={}
            feature_dict['cluster_name']=key_app_feature
            cluster_features=[]
            feature_cluster = []
            for feature_info in cluster_app_features:
                app_feature = feature_info[0]
                app_feature_freq = feature_info[1]
                cluster_features.append({'feature':app_feature,'frequency':app_feature_freq})
                feature_cluster.append(app_feature)
            
            
            feature_dict['cluster_features'] = cluster_features
            #features_clusters.append()
            
            list_app_features.append(feature_dict)
            file.write("%s\n" % (','.join(feature_cluster)))
        
        #print('list of app features')
        #print(list_app_features)
        
        json_output['app_features'] = list_app_features
                
        
        file.close()
        
        return(json_output)
    
    def Merge(self,similarity_threshold=.80):
        self.Merge_Matching_Terms()
        self.Merge_Features_based_WordNet()
        #self.PrintClusteredFeatures(self.feature_cluster)
        self.cluster_features_word_embeddings(similarity_threshold)
        #self.PrintClusteredFeatures(self.feature_cluster_after_embedding)
        return(self.GetAppFeatures_JSONformat())


# In[16]:

# if __name__ == '__main__':
#     app = "422689480"
#     obj_merge_features = Merge_Features(app,MERGE_MODE.DESCRIPTION,nlp)
#     obj_merge_features.Merge()


# In[ ]:





# coding: utf-8

# In[4]:

from enum import Enum

class EVALUATION_TYPE(Enum):
    EXACT=1
    PARTIAL=2


# In[33]:

class Evaluation:
    def __init__(self,true_features,predicted_features,evaluation_type):
        self.true_features=true_features
        self.predicted_features = predicted_features
        self.evaluation_type = evaluation_type
    
    
    def PerformEvaluation(self):
        if self.evaluation_type == EVALUATION_TYPE.EXACT:
            self.ExactEvaluation()
    
    def ExactEvaluation(self):
        
        tp = 0
        fp = 0
        fn = 0
        
        for p_feature in set(self.predicted_features):
            found = False
            matched_feature=""
            for t_feature in set(self.true_features):
                if p_feature.lower() == t_feature.lower():
                    found = True
                    matched_true_feature = t_feature
                    break
            
            if found == True:
                tp =  tp + 1
                print("\'%s\' exactly matched with \'%s\'\n" % (p_feature,matched_true_feature))    
            
            
            if found ==False:
                fp =  fp + 1
        
        
        for t_feature in set(self.true_features):
            found = False
            for p_feature in set(self.predicted_features):
                if p_feature.lower() == t_feature.lower():
                    found = True
                    break
            
            if found == False:
                fn = fn  + 1
                
        print('TP:%d , FP:%d, FN: %d\n' % (tp,fp,fn))
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
        fscore = 2*((precision*recall)/(precision+recall))
        
        print("Precisioin : %.3f, Recall : %.3f, Fscore: %.3f" % (precision,recall,fscore))


# In[35]:

# if __name__ == '__main__':
#     true_features=['natural language parsing', 'reminders', 'week view', 'speak the details of your event', 'see your events', 'see your dated reminders', 'add reminders', 'set dates', 'set times', 'set geofences', 'create reminders with your voice', 'create alerts', 'show event details', "show event's location", 'repeating event options', 'background app updating', 'extended keyboard', 'TextExpander support', 'add new events', 'list events', 'find your events', 'edit reminder', 'push notifications', 'integrates iOS reminders']
#     predicted_features=['facebook events', 'allows reminders', 'new events', 'events fun', 'specific events', 'allows events', 'accessibility support', 'textexpander support', 'including peek', 'week view', 'use dictation', 'calendar services', 'iphone calendar', 'google calendar', 'john gruber', 'reminders fun', 'event show', 'managing schedule', 'stock calendar', 'beautiful week', 'reminder show', 'pure replacement', 'favorite iphone', 'including icloud', 'events reminders', 'allows alerts', 'symbols dates', 'numbers dates', 'details dictation', 'natural language', 'enjoyable way', 'efficient and enjoyable', '3d touch', 'dated reminders', 'apps works', 'handle rest', 'replacement iphone', 'use search', 'event duplicate', 'jim dalrymple', 'winning calendar', 'speak details', 'great ios', 'language parsing', 'ios reminders', 'new features']
    
#     objEvaluation=Evaluation(true_features,predicted_features,EVALUATION_TYPE.EXACT)
#     objEvaluation.PerformEvaluation()


# In[ ]:




# In[ ]:




# build the lightgbm model
import lightgbm as lgb
import torch.nn as tnn
import torch
import numpy as np

from sklearn.metrics import accuracy_score

class lightGBClassifier:
    def __init__(self,learning_rate):
        # self.X_train=X_train
        # self.X_test=X_test
        # self.y_test=y_test
        # self.y_train=y_train
        self.learning_rate= learning_rate
        self.max_depth = -5
        self.random_state = 42



    def classifier(self, X_train, y_train, test = False):
        X_train = X_train.cpu().detach().numpy()
        # X_train.detach().numpy()
        
        y_train = y_train.cpu().detach().numpy()
        # print(X_train)
        # print(y_train)
        # y_train = y_train.detach().numpy()
        
        params={}
        params['learning_rate']=0.05
        params['num_leaves']=130
        params['boosting_type']='gbdt' #GradientBoostingDecisionTree
        params['objective']='multiclass' #Multi-class target feature
        params['metric']='multi_logloss' #metric for multi-class
        params['max_depth']=7
        params['num_class']=6
        params['verbose']=-1
        params['min_data_in_leaf']=1
        # clf = lgb.LGBMClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth, random_state= self.random_state, metric="multi_logloss")
        d_train=lgb.Dataset(X_train, label=y_train)


        # if test == False:
        self.clf = lgb.train(params,d_train)

        # if test == True:
        # y_pred=clf.predict(X_train)
        # y_pred = np.argmax(y_pred,axis=1)



    def predict(self,X_test, y_test):
        y_pred = self.clf.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)

        accuracy = accuracy_score(y_pred, y_test)
        print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy))
        

        # clf.fit(X_train, y_train)

        # predict the results
        # y_pred=clf.predict(self.X_test)

        # JUST FOR TESTING ...
        # y_pred=clf.predict(X_train)
        # print(y_pred)
        # y_pred = np.argmax(y_pred,axis=1)
        
        # print(y_pred)
        # print(y_train)
        
        # view accuracy

        # accuracy = accuracy_score(self.y_pred, self.y_test)
        # print(y_pred)
        # accuracy = accuracy_score(y_pred, y_train)
        # print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy))
        # return(y_pred)

        



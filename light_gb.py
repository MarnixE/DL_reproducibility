# build the lightgbm model
import lightgbm as lgb
import torch.nn as tnn

from sklearn.metrics import accuracy_score

class lightGBClassifier:
    def __init__(self,learning_rate,X_train,X_test,y_train,y_test):
        self.X_train=X_train
        self.X_test=X_test
        self.y_test=y_test
        self.y_train=y_train
        self.learning_rate= learning_rate
        self.max_depth = -5
        self.random_state = 42



    def classifier(self):
        clf = lgb.LGBMClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth, random_state= self.random_state)
        clf.fit(self.X_train, self.y_train)

        # predict the results
        y_pred=clf.predict(self.X_test)

        # view accuracy

        accuracy = accuracy_score(self.y_pred, self.y_test)
        print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
        return(y_pred)

import numpy as np
import pandas as pd
from boruta_py import BorutaPy
from sklearn.feature_selection import RFE, SelectKBest, f_regression, chi2
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# One line explanation:
#   Takes in df, selects the best features for classification using some libraries and outputs the dataframe containing ONLY these features.
#
# Step-by-Step:
#   - Takes in the df (either with the columns labels or without)
#   - Takes in a list of labels (If not included in the df!!!)
#   - You call one of the selector functions e.g. select_by_RFE
#   - You pass in a classifier (if you don't want to use LogisticRegression), and the number of features you want to get back
#   - It does mad math
#   - Finally, outputs a dataframe with the X amount of best features for classifiaction
#
# Additional_note:
#   select_by_boruta function does not care about the number of features - it will output the number features which it thinks are good
#
class assessFeatures:
    def __init__(self, df, labels=None) -> None:
        self.update_df(df, labels)

    def update_df(self, df, labels=None) -> None:
        if 'label' not in df.columns:
            if not labels:
                raise ValueError("Bruuhhh, you don't have labels in your df, AND you didn't give me any. Sad :(")
            self.labels = labels
        else:
            self.labels = df['label']
            df = df.drop(['label'], axis=1)

        if 'user' in df:
            df = df.drop(['user'], axis=1)

        self.features = df.columns
        self.df = df

    def select_by_RFE(self, classifier=LogisticRegression(), num_features=1) -> pd.DataFrame:
        selector = RFE(classifier, n_features_to_select=num_features, step=1)
        selector = selector.fit(self.df, self.labels)

        features_ranking = selector.ranking_
        best_features = []

        for i in range(len(features_ranking)):
            if features_ranking[i] == 1:
                best_features.append(self.features[i])

        return self.df[best_features]
    
    def select_by_ANOVA(self, num_features=1) -> pd.DataFrame:
        selector = SelectKBest(score_func=f_regression, k=num_features)
        selector = selector.fit(self.df, self.labels)

        features_score = pd.DataFrame(selector.scores_)
        features = pd.DataFrame(self.features)
        feature_score = pd.concat([features,features_score],axis=1)

        # Assigning column names
        feature_score.columns = ["Features","F_Score"]
        best_features = list(feature_score.nlargest(num_features,columns="F_Score")['Features'])

        return self.df[best_features]
    
    def select_by_chi2(self, num_features=1) -> pd.DataFrame:
        chi_scores = chi2(abs(self.df), self.labels)
        features_score = pd.Series(chi_scores[0],index = self.features)
        features_score.sort_values(ascending = False , inplace = True)

        best_features = features_score.index[:num_features]

        return self.df[best_features]

    def select_by_boruta(self, num_features=None) -> pd.DataFrame:
        forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=10)
        forest.fit(self.df, self.labels)

        selector = BorutaPy(forest, n_estimators='auto', verbose=0, random_state=42, max_iter=100)
        selector.fit(self.df, self.labels)

        features_ranking = selector.ranking_
        best_features = []

        for i in range(len(features_ranking)):
            if features_ranking[i] == 1:
                best_features.append(self.features[i])

        return self.df[best_features]

    def getScores(self, true, pred):
        fpr, tpr, thresholds = roc_curve(true,pred)
        auc_score = roc_auc_score(true,pred)

        best_threshold = (tpr+(1-fpr)/2)
        best_threshold = thresholds[np.argmax(best_threshold)]

        pred_threshold = (pred >= best_threshold).astype(int)
        precision = precision_score(true, pred_threshold)
        recall = recall_score(true, pred_threshold)
        f1 = f1_score(true, pred_threshold)

        return {"AUC": auc_score, "f1":f1, "recall":recall, "precision":precision}
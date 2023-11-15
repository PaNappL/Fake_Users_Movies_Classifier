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
class selectors:
    def __init__(self, df: pd.DataFrame, labels: list=None) -> None:
        # Set class dataframe and labels
        self.update_df(df, labels)

    def update_df(self, df: pd.DataFrame, labels: list=None) -> None:
        # Check if labels are in dataframe
        if 'label' not in df.columns:
            # If not, check if labels have been passed in as a list
            if not labels:
                # If not raise error
                raise ValueError("Bruuhhh, you don't have labels in your df, AND you didn't give me any. Sad :(")
            # Set labels list to class labels
            self.labels = labels
        else:
            # Set dataframe labels to class labels and drop from database
            self.labels = df['label']
            df = df.drop(['label'], axis=1)

        # Check if user column exists, and remove
        if 'user' in df:
            df = df.drop(['user'], axis=1)

        # Set dataframe column names and dataframe to class features and dataframe
        self.features = df.columns
        self.df = df

    # RFE Feature Selection Algorithm which returns the modified dataframe
    def select_by_RFE(self, classifier=LogisticRegression(), num_features: int=1) -> pd.DataFrame:
        # Initialize and train algorithm
        selector = RFE(classifier, n_features_to_select=num_features, step=1)
        selector = selector.fit(self.df, self.labels)

        # Retrieve features ranking
        features_ranking = selector.ranking_
        # Create empty list for storing feature names
        best_features = []

        # Loop through features and select best features
        for i in range(len(features_ranking)):
            # Check if feature ranked as 1
            if features_ranking[i] == 1:
                # If yes, append to list
                best_features.append(self.features[i])

        # Return dataframe with features = best_features
        return self.df[best_features]
    
    # ANOVA Feature Selection Algorithm which returns the modified dataframe
    def select_by_ANOVA(self, num_features: int=1) -> pd.DataFrame:
        # Initialize and train algorithm
        selector = SelectKBest(score_func=f_regression, k=num_features)
        selector = selector.fit(self.df, self.labels)

        # Retrieve features scores and feature names from ANOVA
        features_score = pd.DataFrame(selector.scores_)
        features = pd.DataFrame(self.features)
        # Combine feature names with their respective score
        feature_score = pd.concat([features,features_score],axis=1)

        # Assigning column names
        feature_score.columns = ["Features","F_Score"]
        # Retrieve num_features amount of features as list
        best_features = list(feature_score.nlargest(num_features,columns="F_Score")['Features'])

        # Return dataframe with features = best_features
        return self.df[best_features]
    
    # chi2 Feature Selection Algorithm which returns the modified dataframe
    def select_by_chi2(self, num_features: int=1) -> pd.DataFrame:
        # Initialize and train algorithm
        chi_scores = chi2(abs(self.df), self.labels)

        # Retrieve features scores and sort by score
        features_score = pd.Series(chi_scores[0],index = self.features)
        features_score.sort_values(ascending = False , inplace = True)

        # Retrieve num_features amount of features as list
        best_features = features_score.index[:num_features]

        # Return dataframe with features = best_features
        return self.df[best_features]

    # boruta Feature Selection Algorithm which returns the modified dataframe
    def select_by_boruta(self, num_features: int=None) -> pd.DataFrame:
        # Initialize and train Random Forest Classifier
        forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=10)
        forest.fit(self.df, self.labels)

        # Initialize and train algorithm
        selector = BorutaPy(forest, n_estimators='auto', verbose=0, random_state=42, max_iter=100)
        selector.fit(self.df, self.labels)

        # Retrieve features ranking
        features_ranking = selector.ranking_
        # Create empty list for storing feature names
        best_features = []

        # Loop through features and select best features
        for i in range(len(features_ranking)):
            # Check if feature ranked as 1
            if features_ranking[i] == 1:
                # If yes, append to list
                best_features.append(self.features[i])

        # Return dataframe with features = best_features
        return self.df[best_features]

    # Calculate prediction scores and return scores in dictionary
    def getScores(true: list, pred: list) -> dict:
        # Calculate ROC curve and AUC score of prediction
        fpr, tpr, thresholds = roc_curve(true,pred)
        auc_score = roc_auc_score(true, pred)

        # Find the most optimal threshold using the roc curve
        best_threshold = (tpr+(1-fpr)/2)
        best_threshold = thresholds[np.argmax(best_threshold)]
        # Apply threshold to predictions
        pred = (pred >= best_threshold).astype(int)

        # Calculate precision, recall and f1 scores using the thresholded predictions
        precision = precision_score(true, pred)
        recall = recall_score(true, pred)
        f1 = f1_score(true, pred)

        # Return the scores inside a dictionary
        return {"Thresh":best_threshold, "Precision":precision, "Recall":recall, "F1_Score":f1, "AUC":auc_score}
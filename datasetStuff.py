import numpy as np
import pandas as pd
import math
from boruta_py import BorutaPy
from scipy.stats import skew, kurtosis, entropy
from sklearn.feature_selection import RFE, SelectKBest, f_regression, chi2, mutual_info_classif
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Sorry for lack of comments, I kinda not bothered
# Lazy af
# But but!
# Hope this helps
#
#
# Quick intro: woo!
# There are two classes here:
#   expandDataset
#   assessFeatures
#
# expandDataset:
#   You bippyty boppity the loaded data from the npz files (the ones prof sent)
#   And the code below does the following:
#       Performs the same stuff prof does (or gave us) at the beginning of the jupyter notebook
#       Creates the features that were created by someone (Sorry! I got no idea exactly who thought and masterminded these features; if you let me know I'll adjust this comment to make you famous! ;D )
#       Creates some MORE features that I kinda half-arsedly created from other features just for the sake of it (maybe will help maybe not)
#
#   As some may not want my features cuz they wack as hell there is a param below (## more_features ##) which you can modify (i.e. pass a variable to change it)
#       --By default it is set to False, so that it won't add it
#
#   Unfortunately you can't return the df directly from the init function, so to get back the converted df...
#   ... you have to call the getDataset() function
#
#   As a quick demo, you can do:
#   
#   data=np.load("first_batch_with_labels_likes.npz")
#   df = expandDataset(data, more_features=True).getDataset()
#   
#   And Done! That's all folks! You got a dataframe!
#   Nice! Super hot, super sexy!
#
# Extra info:
#   I know that you may have created more features, so if you want, you can do either of the two (totally optional if you even want to do this):
#       - Add into one of the funcitons: createHellaFeatures OR createHellaMoreFeatures
#       - Create a seperate function to do this (although remember to include it into the init function)
#
#
#
# Explanation of assessFeatures beloooo! :)
#
class expandDataset:

    # NOTE TO PROF:
    # The code is hella rushed - not commented enough, code was added from other files, so that all features can be seen here in one file
    # This will be cleaned up and made nicer in final submission :)

    def __init__(self, data, more_features=False) -> None:
        df, labels = self.simpleDataRetrieval(data)
        userItem = self.generateUserItem(df)

        # Group by 'item' and calculate the sum of 'rating' for each item
        item_ratings = df.groupby('item')['rating'].sum().to_dict()
        categories = self.categorise_popularity(item_ratings)

        # Create Sparse Matrix
        sm = self.generateSparseMatrix(df)
        # Create items features
        items_feat = stuff().doStuff(df)
        # Create features created at the beginning of project
        feats = self.createHellaFeatures(df, labels, userItem, categories)

        if more_features:
            feats = self.createHellaMoreFeatures(feats)

        feats = feats.drop(list(set(sm.columns)&set(feats.columns)),axis=1)
        items_feat = items_feat.drop(list(set(items_feat.columns)&set(sm.columns)),axis=1)

        sm = sm.merge(feats, left_index=True, right_index=True)
        df = sm.merge(items_feat, left_index=True, right_index=True)

        if 'user_y' in df.columns:
            df = df.drop(['user_y'],axis=1)

        self.df = df

    def getDataset(self) -> pd.DataFrame:
        return self.df

    def simpleDataRetrieval(self, data) -> pd.DataFrame:
        X=data["X"]
        XX=pd.DataFrame(X)
        XX.rename(columns={0:"user",1:"item",2:"rating"},inplace=True)

        try:
            y=data["y"]
            yy=pd.DataFrame(y)
            yy.rename(columns={0:"user",1:"label"},inplace=True)
        except:
            yy = None
            
        return XX, yy
    
    def generateUserItem(self,df) -> dict:
        XX = df.sort_values(by=["item"], ascending=True)
        dictItem = self.generateDictItem(XX)

        XX = df.sort_values(by=["user"], ascending=True)
        curr_user = -1

        userItem = {}
        for index, row in XX.iterrows():
            if (row["user"] != curr_user):
                curr_user +=1
                userItem[curr_user] = 0
                
            mostPopRating = dictItem[row["item"]] 
            
            if(mostPopRating == row["rating"]):
                userItem[curr_user] = userItem[curr_user] + 1
        return userItem
    
    def generateSparseMatrix(self, XX) -> dict:
        sm = pd.DataFrame(0, index=range(len(set(XX['user']))), columns=range(XX['item'].max()+1))
        sm.index = list(set(XX.user))
        sm.sort_index()

        XX = XX.sort_values(by=["item"], ascending=True)
        dictItem = {}
        for i, row in XX.iterrows():
            rating = row["rating"]
            item = row["item"]

            try:
                keyValue = dictItem[item]
            except KeyError:
                keyValue = [0,0,0]
            if(rating == 1):
                keyValue[0] = keyValue[0] + 1
                dictItem[item] = keyValue

            elif(rating == 0):
                keyValue[1] = keyValue[1] + 1
                dictItem[item] = keyValue

            elif(rating == -1):
                keyValue[2] = keyValue[2] + 1
                dictItem[item] = keyValue
                
        for i in dictItem:
            itemValue = dictItem[i]
            largestIdx = itemValue.index(max(itemValue))
            if(largestIdx == 0):
                dictItem[i] = 1
            if(largestIdx == 1):
                dictItem[i] = 0
            if(largestIdx == 2):
                dictItem[i] = -1

        XX = XX.sort_values(by=["user"], ascending=True)
        curr_user = -1
        userItem = {}
        for index, row in XX.iterrows():
            if (row["user"] != curr_user):
                curr_user = row["user"]
                userItem[curr_user] = 0

            mostPopRating = dictItem[row["item"]]

            if(mostPopRating == row["rating"]):
                userItem[curr_user] = userItem[curr_user] + 1

        firstKey = list(userItem.keys())[0]

        userItemCol = pd.DataFrame(userItem.values())
        userItemCol.index = pd.RangeIndex(firstKey,firstKey + len(userItemCol))

        # Ratings will now be, 0 -> didn't rate, 1 -> dislike, 2 -> meh, 3 -> like
        XX["rating"] = XX["rating"].replace(1, 3)
        XX["rating"] = XX["rating"].replace(0, 2)
        XX["rating"] = XX["rating"].replace(-1, 1)
        curr_user = -1

        for index, row in XX.iterrows():
            if (row["user"] != curr_user):
                curr_user = row["user"]
            sm.loc[int(curr_user), int(row["item"])] = row["rating"]

        sm = sm.sort_index()

        return sm

    def generateDictItem(self,df) -> dict:
        dictItem = {}
        # print(df)
        for i, row in df.iterrows():
            rating = row["rating"]
            item = row["item"]
            
            try:
                keyValue = dictItem[item]
            except KeyError:
                keyValue = [0,0,0]
            if(rating == 1):
                keyValue[0] = keyValue[0] + 1
                dictItem[item] = keyValue
                
            elif(rating == 0):
                keyValue[1] = keyValue[1] + 1
                dictItem[item] = keyValue
                
            elif(rating == -1):
                keyValue[2] = keyValue[2] + 1
                dictItem[item] = keyValue

        for i in dictItem:
            itemValue = dictItem[i]
            largestIdx = itemValue.index(max(itemValue))
            if(largestIdx == 0): 
                dictItem[i] = 1
            if(largestIdx == 1): 
                dictItem[i] = 0
            if(largestIdx == 2):
                dictItem[i] = -1
        return dictItem

    def categorise_popularity(self, item_ratings):
        categories = {}
        for item_id, rating in item_ratings.items():
            # 1: Popular
            # 0: Neutral Popularity
            # -1: Unpopular
            if rating > 50:
                categories[item_id] = 1
            elif rating > 0:
                categories[item_id] = 0
            else:
                categories[item_id] = -1
        return categories

    def aggregate_user_ratings(self, df, categories):
        # Pivot the DataFrame to have users as rows, items as columns, and ratings as values
        pivot_df = df.pivot_table(index='user', columns='item', values='rating', fill_value=0)

        # For all users, each element represents one user
        num_popular_items_liked = []
        num_unpopular_items_liked = []
        num_popular_items_disliked = []
        num_unpopular_items_disliked = []

        # Iterate over the rows (users) in the pivot table
        for index, row in pivot_df.iterrows(): # for each user
            num_pop_items_liked = 0
            num_unpop_items_liked = 0
            num_pop_items_disliked = 0
            num_unpop_items_disliked = 0

            for i, rating in enumerate(row):
                # Get item's popularity
                item_popularity = categories.get(i)

                # Add the count to the correct category
                if (item_popularity == 1 and rating == 1):
                    num_pop_items_liked += 1
                elif (item_popularity == -1 and rating == 1):
                    num_unpop_items_liked += 1
                elif (item_popularity == 1 and rating == -1):
                    num_pop_items_disliked += 1
                else:
                    num_unpop_items_disliked += 1

            # After iterating all items rated by a user, add the final results
            num_popular_items_liked.append(num_pop_items_liked)
            num_unpopular_items_liked.append(num_unpop_items_liked)
            num_popular_items_disliked.append(num_pop_items_disliked)
            num_unpopular_items_disliked.append(num_unpop_items_disliked)
        return num_popular_items_liked, num_unpopular_items_liked, num_popular_items_disliked, num_unpopular_items_disliked

    # Testing
    def createHellaFeatures(self, df, labels, userItem, categories):
        num_pop_liked, num_unpop_liked, num_pop_disliked, num_unpop_disliked = self.aggregate_user_ratings(df, categories)

        # Grouping by user and creating aggregated features
        df_grouped = df.groupby('user').agg(
            average_rating=('rating', 'mean'),
            total_interactions=('rating', 'size'),
            likes=('rating', lambda x: (x == 1).sum()),
            dislikes=('rating', lambda x: (x == -1).sum()),
            meh=('rating', lambda x: (x == 0).sum()),
            neutral_ratings=('rating', lambda x: (x == 0).sum()),
            entropy=('rating', lambda x: entropy(x.value_counts(normalize=True))),

            item_mean=('item', 'mean'),
            item_variance=('item', 'var'),
            skewness=('item', skew),
            kurtosis=('item', kurtosis),
        )
        df_grouped['likes_ratio'] = df_grouped['likes'] / df_grouped['total_interactions']
        df_grouped['dislikes_ratio'] = df_grouped['dislikes'] / df_grouped['total_interactions']
        df_grouped['interaction_balance'] = df_grouped['likes'] - df_grouped['dislikes']
        df_grouped['neutral_ratio'] = df_grouped['neutral_ratings'] / df_grouped['total_interactions']
        df_grouped['balance_ratio'] = df_grouped['interaction_balance'] / df_grouped['total_interactions']
        df_grouped['mean'] = df_grouped['likes'] + df_grouped['dislikes'] + df_grouped['meh'] / 3
        df_grouped['std'] = df_grouped[['likes', 'dislikes', 'meh']].std(axis=1)
        df_grouped['cv'] = df_grouped['std']/df_grouped['mean'] * 100
        df_grouped['followed majority'] = pd.DataFrame(userItem.values())
        df_grouped['followed majority %'] = df_grouped['followed majority'] / df_grouped['total_interactions']
        df_grouped['rating_val'] = (df_grouped['dislikes'] + df_grouped['meh']*2 + df_grouped['likes']*3)/df_grouped['total_interactions']**2
        df_grouped['no_rating'] = len(df['item'].unique())-df_grouped['total_interactions']
        df_grouped["num_pop_liked"] = num_pop_liked
        df_grouped["num_unpop_liked"] = num_unpop_liked
        df_grouped["num_pop_disliked"] = num_unpop_disliked
        df_grouped["num_unpop_disliked"] = num_unpop_disliked

        # Merging with labels to create a single DataFrame
        if type(labels) == pd.DataFrame:
            df_final = df_grouped.merge(labels, left_index=True, right_on='user')
        else:
            df_final = df_grouped

        return df_final

    def createHellaMoreFeatures(self, df):
        new_feature_names = ['ld_ratio','dm_ratio','lm_ratio','pythagoras_ldm','pythagoras_ratios','log_ldm','pyth_log_ratios','follow_rat_val', 'plr_frv', 'cv_foll', 'log_ratios', 'rating_ratio','pyth_log_ldm_ratios']
        new_features = [[] for i in range(len(new_feature_names))]

        for index, row in df.iterrows():
            no_zero = lambda x: 1 if x == 0 else x
            not_inf = lambda x: 0 if x == np.inf else x

            likes = no_zero(int(row['likes']))
            dislikes = no_zero(int(row['dislikes']))
            meh = no_zero(int(row['meh']))

            feature_1 = not_inf(np.max([likes,dislikes])/np.min([likes,dislikes])) # not rlly
            feature_2 = not_inf(np.max([dislikes,meh])/np.min([dislikes, meh])) # not rlly
            feature_3 = not_inf(np.max([likes,meh])/np.min([likes,meh])) # Not rlly
            feature_4 = math.sqrt(likes**2+dislikes**2+meh**2)
            feature_5 = math.sqrt(feature_1**2+feature_2**2+feature_3**2)
            feature_6 = math.log(likes*dislikes*meh)
            feature_7 = feature_6*feature_5
            feature_8 = row['followed majority %']*row['rating_val']
            feature_9 = feature_7*feature_8
            feature_10 = row['cv']*row['followed majority %'] # great
            feature_11 = math.log(feature_1*feature_2*feature_3)
            feature_12 = row['total_interactions']/(row['no_rating']+row['total_interactions'])
            feature_13 = math.log(no_zero(feature_4*feature_5*feature_6*feature_11))*feature_8

            features = [feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8,feature_9,feature_10,feature_11,feature_12,feature_13]

            for i in range(len(new_features)):
                new_features[i].append(features[i])

        for i in range(len(new_features)):
            df[new_feature_names[i]] = new_features[i]

        return df

class stuff:
    def __init__(self):
        return
    
    def doStuff(self, df):
        XX = df.sort_values(by=["item"], ascending=True)
        XX = XX.drop(['user'],axis=1)
        XX = XX.groupby(by='item').aggregate('mean')

        a = []
        for i in df['user'].unique():
            items = list(df.sort_values(by='item')['item'].loc[df['user']==i])
            av_diff=0
            for j in range(0,len(items)-1):
                diff = items[j+1]-items[j]
                av_diff += diff
            av_diff /= len(items)
            item_comp = {'user':i,'items_diff':av_diff}
            a.append(item_comp)
        user_items_diff = pd.DataFrame(a)
        user_items_diff = user_items_diff.sort_values(by='user').reset_index(drop=True)

        f = lambda row: -1 if row < -0.5 else 0
        XX['mode'] = XX['rating'].apply(lambda row: 1 if row > 0.5 else f(row))

        user_item_comp = []
        for i in df['user'].unique():
            user_df = df.loc[df['user']==i]
            item_comp = {'user':i,'modal_ratings':0,'non_modal_ratings':0,'rating_diff':[]}
            for j in user_df['item']:
                user_rating = user_df['rating'].loc[user_df['item']==j].iloc[0]
                item_mode = XX['mode'][j]

                if user_rating == item_mode:
                    item_comp['modal_ratings'] += 1
                else:
                    item_comp['non_modal_ratings'] += 1

                item_comp['rating_diff'].append(abs(item_mode-user_rating))
            user_item_comp.append(item_comp)

        df_new = pd.DataFrame(user_item_comp).sort_values(by='user').reset_index(drop=True)
        df_new['diff_std'] = df_new['rating_diff'].apply(lambda row: np.std(row))
        df_new['diff_mean'] = df_new['rating_diff'].apply(lambda row: np.mean(row))

        mode_ratings_diff = []
        for index,row in df_new.iterrows():
            x = row['modal_ratings']
            y = row['non_modal_ratings']
            diff = min(x,y)/max(x,y)
            mode_ratings_diff.append(diff)
        df_new['mode_ratings_diff'] = mode_ratings_diff
        df_new = df_new.sort_values(by='user').reset_index(drop=True)
        df_new = df_new.join(user_items_diff['items_diff'])

        df_new = df_new.drop(['rating_diff'],axis=1)
        df_new.index = df_new['user']

        return df_new


# Now, something maybe a bit more interesting
# assessFeatures!!!
#
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

    #
    #
    # Not included as will later finish it and make good
    #
    #

    # def select_by_mutualInfo(self, threshold=0.02):
    #     features_ranking = mutual_info_classif(self.df, self.labels)
        
    #     best_features = []

    #     for i in range(len(features_ranking)):
    #         if features_ranking[i] >= threshold:
    #             best_features.append(self.features[i])

    #     return self.df[best_features]
# The end :)
import pandas as pd
import numpy as np
import math

class feature_gen:
    # Retrieve data from path and create data features
    def retrieveAndGenerate(self, path: str) -> pd.DataFrame:
        # Retrieve data from file path
        df, yy = self.extractDataFromNPZ(path)
        # Generate features from data
        df = self.generateFeatures(df, yy)
        # Generate new features for week 2
        df = self.generateFeatures2(df)

        # Return generated features
        return df

    # Load and extract data into pd.DataFrane from file
    def extractDataFromNPZ(self, path: str) -> pd.DataFrame:
        # Load data from provided file path
        data=np.load(path)

        # Extract {user : item : rating} pairs into pd.DataFrame
        X=data["X"]
        XX=pd.DataFrame(X)
        XX.rename(columns={0:"user",1:"item",2:"rating"},inplace=True)

        # Attempt to extract "y" data from loaded data 
        try:
            # If y is present in data, extract {user : label} pairs into pd.DataFrame
            y=data["y"]
            yy=pd.DataFrame(y)
            yy.rename(columns={0:"user",1:"label"},inplace=True)
        except:
            # If y is not in data, set yy to None
            yy = None
            
        # Return XX and yy
        return XX, yy
    
    # Generate features from input data
    def generateFeatures(self, df: pd.DataFrame, yy: pd.DataFrame) -> pd.DataFrame:
        # Generate dictionary of user top ratings count
        userItemTopRatingsCount = self.__genUserTopRatingsCount(df)
        # Get amount of unique items in data
        items_amount = len(df['item'].unique())

        # Group data by user value
        df_final = df.groupby('user').agg(
            # Aggregate grouped data to calculate:
            # Average Rating, Total Interactions, Amount of Likes, Amount of Dislikes, Amount of Neutral Ratings
            average_rating=('rating', 'mean'),
            total_interactions=('rating', 'size'),
            likes=('rating', lambda x: (x == 1).sum()),
            dislikes=('rating', lambda x: (x == -1).sum()),
            neutral=('rating', lambda x: (x == 0).sum())
        )

        # Calculate likes, dilikes and neutral ratings ratios by dividing the respective values by the total amount of interactions
        df_final['likes_ratio'] = df_final['likes'] / df_final['total_interactions']
        df_final['dislikes_ratio'] = df_final['dislikes'] / df_final['total_interactions']
        df_final['neutral_ratio'] = df_final['neutral'] / df_final['total_interactions']

        # Calculate user interaction_balance and balance_ratio
        df_final['interaction_balance'] = df_final['likes'] - df_final['dislikes']
        df_final['balance_ratio'] = df_final['interaction_balance'] / df_final['total_interactions']

        # Calculate the mean, std and cv values for user ratings
        df_final['mean'] = df_final['likes'] + df_final['dislikes'] + df_final['neutral'] / 3
        df_final['std'] = df_final[['likes', 'dislikes', 'neutral']].std(axis=1)
        df_final['cv'] = df_final['std']/df_final['mean'] * 100

        # Assign the amount of user top ratings count to "followed majority" and calculate "followed majority %"
        df_final['followed majority'] = pd.DataFrame(userItemTopRatingsCount.values())
        df_final['followed majority %'] = df_final['followed majority'] / df_final['total_interactions']

        ## --------------------------------------------- NEW --------------------------------------------- ##

        # Average rating value for all ratings
        df_final['rating_val'] = (df_final['dislikes'] + df_final['neutral']*2 + df_final['likes']*3)/df_final['total_interactions']**2

        ## ----------------------------------------------------------------------------------------------- ##

        # Calculate amount of no ratings provided
        df_final['no_rating'] = items_amount - df_final['total_interactions']

        if type(yy) == pd.DataFrame:
            # Merging with labels to create a single DataFrame
            df_final = df_final.merge(yy, left_index=True, right_on='user')

        # Return final dataframe with calculated data features
        return df_final

    # Create a dictionary of user : top ratings count
    # Explanation:
    #       top ratings count: Amount of user ratings which are the top rating for the given items
    def __genUserTopRatingsCount(self, df: pd.DataFrame) -> dict:

        # Generate dictionary of top ratings for item
        userItemTopRatings = self.__genItemTopRatings(df)

        # Sort data by user value
        XX = df.sort_values(by=['user'], ascending=True)
        # Create and initialize a dictionary of user and top ratings count
        userTopRatingsCount = {user:0 for user in XX['user']}

        # Iterate through user ratings and increment user top ratings count by 1
        for index, row in XX.iterrows():
            # Retrieve the top ratings for following item
            mostPopRating = userItemTopRatings[row['item']] 

            # Check if the user rating is a top rating
            if(row['rating'] in mostPopRating):
                # If yes, increment top ratings count by 1
                userTopRatingsCount[row['user']] += 1

        # Return userTopRatingsCount
        return userTopRatingsCount

    # Create a dictionary of item : top rating pairs
    # Example:
    #       0 : [1]
    # Explanation:
    #       Item's 0 most popular rating is 1
    def __genItemTopRatings(self, df: pd.DataFrame) -> dict:

        # Sort data by item value
        XX = df.sort_values(by=["item"], ascending=True)
        # Create and initialize a dictionary of items and ratings count
        itemTopRatings = {item:{-1:0, 0:0, 1:0} for item in XX['item']}

        # Iterate through all ratings for items and increment the corresponding item : rating count by 1
        for i, row in XX.iterrows():
            rating = row["rating"]
            item = row["item"]

            itemTopRatings[item][rating] += 1

        # Iterate through all items and assign top ratings
        for item, ratings in itemTopRatings.items():
            # Find largest value in ratings
            max_value = max(ratings.values())
            # Retrieve keys with the largest value
            top_ratings = [key for key, value in ratings.items() if value == max_value]
            # Assign most popular ratings to item in dictionary
            itemTopRatings[item] = top_ratings

        # Return itemTopRatings
        return itemTopRatings
    
    ## --------------------------------------------- NEW --------------------------------------------- ##
    
    def generateFeatures2(self, df: pd.DataFrame) -> pd.DataFrame:
        # Create lists of feature names and empty array for feature values
        new_feature_names = ['ld_ratio','dm_ratio','lm_ratio','pythagoras_ldm','pythagoras_ratios','log_ldm','pyth_log_ratios','follow_rat_val', 'plr_frv', 'cv_foll', 'log_ratios', 'rating_ratio','pyth_log_ldm_ratios']
        placeholder_values = [0 for i in range(len(new_feature_names))]

        df[new_feature_names] = placeholder_values

        for index, row in df.iterrows():
            no_zero = lambda x: 1 if x == 0 else x
            not_inf = lambda x: 0 if x == np.inf else x

            likes = no_zero(int(row['likes']))
            dislikes = no_zero(int(row['dislikes']))
            neutral = no_zero(int(row['neutral']))

            feature_1 = not_inf(np.max([likes,dislikes])/np.min([likes,dislikes])) # not rlly
            feature_2 = not_inf(np.max([dislikes,neutral])/np.min([dislikes, neutral])) # not rlly
            feature_3 = not_inf(np.max([likes,neutral])/np.min([likes,neutral])) # Not rlly
            feature_4 = math.sqrt(likes**2+dislikes**2+neutral**2)
            feature_5 = math.sqrt(feature_1**2+feature_2**2+feature_3**2)
            feature_6 = math.log(likes*dislikes*neutral)
            feature_7 = feature_6*feature_5
            feature_8 = row['followed majority %']*row['rating_val']
            feature_9 = feature_7*feature_8
            feature_10 = row['cv']*row['followed majority %'] # great
            feature_11 = math.log(feature_1*feature_2*feature_3)
            feature_12 = row['total_interactions']/(row['no_rating']+row['total_interactions'])
            feature_13 = math.log(no_zero(feature_4*feature_5*feature_6*feature_11))*feature_8

            features = [feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8,feature_9,feature_10,feature_11,feature_12,feature_13]

            df.loc.__setitem__((index, new_feature_names), features)

        return df
    
    ## ----------------------------------------------------------------------------------------------- ##
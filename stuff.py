import pandas as pd
import numpy as np

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

        return df_new

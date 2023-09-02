from fastapi import FastAPI

app = FastAPI()

import pandas as pd
import numpy as np
import ast
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

userdata_games = pd.read_csv('./API_csv/userdata_games.csv', sep=',')
userdata_items = pd.read_csv('./API_csv/userdata_items.csv', sep=',')
userdata_reviews = pd.read_csv('./API_csv/userdata_reviews.csv', sep=',')

#userdata_items['item_id'] = userdata_items['item_id'].apply(ast.literal_eval)

def convert_to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return value

userdata_games['price'] = userdata_games['price'].apply(convert_to_float)

def convert_to_list(value):
    try:
        return list(value)
    except (ValueError, TypeError):
        return value
    
#userdata_items['item_id'] = userdata_items['item_id'].apply(convert_to_list)



countreviews_date_reviews = pd.read_csv('./API_csv/countreviews_date_reviews.csv', sep=',')
countreviews_date_reviews = countreviews_date_reviews.dropna(subset=['date'])
countreviews_date_reviews['date'] = pd.to_datetime(countreviews_date_reviews['date']).dt.date


genre_genre_rank = pd.read_csv('./API_csv/genre_genre_rank.csv', sep=',')



userforgenre_playtime_rank = pd.read_csv('./API_csv/userforgenre_playtime_rank.csv', sep=',')



developer_merged_developer = pd.read_csv('./API_csv/developer_merged_developer.csv', sep=',')



sentiment_analysis_sentiment_analysis_developer = pd.read_csv('./API_csv/sentiment_analysis_sentiment_analysis_developer.csv', sep=',')



#ML_games_model = pd.read_csv('./API_csv/ML_games_model.csv', sep=',')

#ML_games_selected2 = pd.read_csv('./API_csv/ML_games_selected2.csv', sep=',', header=None)
#ML_games_selected2 = ML_games_selected2[0]

#ML_games_u_p = pd.read_csv('./API_csv/ML_games_u_p.csv', sep=',')

#vectorizer = CountVectorizer()
#vectorized = vectorizer.fit_transform(ML_games_selected2)

#similarities = cosine_similarity(vectorized)

#games_vect = pd.DataFrame(data = similarities, columns=ML_games_model['id'], index=ML_games_model['id']).reset_index()

#games_concat = pd.concat([ML_games_u_p, games_vect], axis=1)

@app.get("/")
def root():
    return {"message": "Hello Mundo"}

@app.get("/user_data/")
async def userdata(user_id: str):
    try:
        target_user_id = str(user_id)
    except ValueError:
        return {"error": "Invalid user_id"}
    total_price = 0

    for index, row in userdata_items.iterrows():
        if target_user_id in row['user_id']:
            row['item_id'] = ast.literal_eval(row['item_id'])
            for item_id in row['item_id']:
                game_row = userdata_games[userdata_games['id']==float(item_id)]
                if not game_row.empty:
                    price = game_row['price'].values[0]
                    if price is not None:
                        if isinstance(price, (int, float)):
                            total_price += float(price)

    row_reviews = userdata_reviews[userdata_reviews['user_id'] == target_user_id]
    pct_recommend = row_reviews['pct_recommend'].values[0]

    row_items = userdata_items[userdata_items['user_id'] == target_user_id]
    items_count = int(row_items['items_count'].values[0])

    result_dict = {
                    'user': target_user_id,
                    'amount_spent': round(total_price, 2),
                    'recommendation_pct': round(pct_recommend, 2),
                    'items_count': items_count
    }

    return result_dict

@app.get("/countreviews/")
async def countreviews(start_date: str, end_date: str):
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    filtered_dates = countreviews_date_reviews[
        (countreviews_date_reviews['date'] >= start_date) & 
        (countreviews_date_reviews['date'] <= end_date)
    ]
    
    unique_user_count = filtered_dates['user_id'].nunique()
    unique_pct_recommend = filtered_dates['pct_recommend'].mean()

    result_dict = {
        'start_date': start_date.strftime("%Y-%m-%d"), 
        'end_date': end_date.strftime("%Y-%m-%d"),      
        'users_count': unique_user_count,
        'recommend_pct': unique_pct_recommend
    }

    return result_dict

@app.get("/genre/")
async def genre(genre: str):
    genre = genre.lower().capitalize()
    row = genre_genre_rank[genre_genre_rank['genres'] == genre]
    
    try:
        playtime_rank = int(row['playtime_rank'].values[0])
    except:
        playtime_rank = row['playtime_rank'].values[0]

    if not isinstance(playtime_rank, (int, float)):
        return "Género no encontrado"

    result_dict = {
                    'genre': genre,
                    'playtime_rank': playtime_rank
    }
    
    return result_dict

@app.get("/userforgenre/")
async def userforgenre(genre: str):
    genre = genre.lower().capitalize()
    genre_data = userforgenre_playtime_rank[userforgenre_playtime_rank['genres'] == genre].head(5)

    result_dict = genre_data.to_dict(orient='records')

    return result_dict

@app.get("/developer/")
async def developer(developer: str):
    result_df = developer_merged_developer[developer_merged_developer['developer'] == developer].reset_index()

    result_dict = result_df[['year', 'pct_free']].to_dict(orient='records')
    return result_dict

@app.get("/sentiment_analysis/")
async def sentiment_analysis(year: int):
    
    year_df = sentiment_analysis_sentiment_analysis_developer[sentiment_analysis_sentiment_analysis_developer['release_date'] == year]
    
    counts = year_df['sentiment_analysis'].value_counts().to_dict()
    
    for sentiment in [0, 1, 2]:
        if sentiment not in counts:
            counts[sentiment] = 0
    
    sentiment_dict = {
        'Negative': counts[0],
        'Neutral': counts[1],
        'Positive': counts[2]
    }
    
    return sentiment_dict

#@app.get("/game_recommendation/")
#async def game_recommendation(id: int):
 #   try:
  #      print('Wait a minute and you will see our recommendations. Enjoy! ')
   #     recommendations = pd.DataFrame(games_concat.nlargest(6,id)['id'])
    #    recommendations = recommendations[recommendations['id']!=id]
     #   title = pd.DataFrame(games_concat.nlargest(6,id)['title'])
      #  price = pd.DataFrame(games_concat.nlargest(6,id)['price'])
       # url = pd.DataFrame(games_concat.nlargest(6,id)['url'])
        #result2 = pd.concat([recommendations,title, price[1:11],url[1:]], axis = 1)
        #result2 = result2.to_dict()
        #return result2
    #except:
     #   print('Sorry, we can not find a suitable match. Try a different game! ')

import nest_asyncio
import uvicorn

if __name__ == '__main__':
    nest_asyncio.apply()
    uvicorn.run(app)

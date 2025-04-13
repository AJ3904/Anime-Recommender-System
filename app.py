import pickle
import streamlit as st
import numpy as np

st.header("Anime Recommender System")

model = pickle.load(open("models/model.pkl", "rb"))
anime_name = pickle.load(open("artifacts/anime_name.pkl", "rb"))
anime_information = pickle.load(open("artifacts/final_information.pkl", "rb"))
anime_pivot = pickle.load(open("artifacts/pivot_table.pkl", "rb"))

selected_anime = st.selectbox("Enter Anime Name", anime_name)

def fetch_rating_and_synopsis(suggestions):
    anime_titles = []
    ids_index = []
    ratings = []
    synopsises = []

    for anime_id in suggestions:
        anime_titles.append(anime_pivot.index[anime_id])

    for name in anime_titles[0]:
        id = np.where(anime_information['Name'] == name)[0][0]
        ids_index.append(id)

    for idx in ids_index:
        rating = anime_information.iloc[idx]['Rating Score']
        synopsis = anime_information.iloc[idx]['Synopsis']
        synopsises.append(synopsis)
        ratings.append(rating)
    
    return ratings, synopsises

def clean_title(title):
    title = title.lower()
    title = title.replace('season', '')
    title = title.replace('ova', '')
    title = title.replace('movie', '')
    title = title.replace('Movie', '')
    title = title.replace('2nd', '')
    title = title.replace('3rd', '')
    title = title.replace('part', '')
    title = title.replace('ii', '')
    title = title.replace('iii', '')
    title = title.replace('iv', '')
    return title.strip()

def recommend_anime(selected_anime):
    recommendations = []
    anime_id = np.where(anime_pivot.index == selected_anime)[0][0]
    distance, suggestions = model.kneighbors(anime_pivot.iloc[anime_id, :].values.reshape(1, -1), n_neighbors=20) # Increase neighbors a bit

    rating_score, synopsis = fetch_rating_and_synopsis(suggestions)

    selected_main_title = clean_title(selected_anime)

    for i in range(len(suggestions)):
        anime_list = anime_pivot.index[suggestions[i]]
        for anime in anime_list:
            if clean_title(anime) != selected_main_title:
                recommendations.append(anime)
    
    # Also filter rating_score and synopsis to match
    final_ratings = []
    final_synopses = []
    for i in range(len(recommendations)):
        id = np.where(anime_information['Name'] == recommendations[i])[0][0]
        final_ratings.append(anime_information.iloc[id]['Rating Score'])
        final_synopses.append(anime_information.iloc[id]['Synopsis'])

    return recommendations[:8], final_ratings[:8], final_synopses[:8]

if st.button('Show recommendations'):
    recommendations, rating_score, synopsis = recommend_anime(selected_anime)
    st.subheader("Here are your recommendations:")
    for i in range(len(recommendations)):
        st.markdown(f"**{i+1}. {recommendations[i]}**")
        st.markdown(f"**Rating:** {rating_score[i]}")
        st.markdown(f"**Synopsis:** {synopsis[i]}")
        st.markdown("---")
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import mysql.connector

# Connect to MySQL
def connect_to_mysql():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="youtube" 
    )

def fetch_data_from_mysql():
    connection = connect_to_mysql()
    mycursor = connection.cursor()
    
    query = '''
    SELECT video_id, channel_id, video_title, video_thumbnails, video_description, channelTitle, 
           video_tags, video_duration, video_viewCount, video_likeCount, video_commentCount
    FROM youtube.video_table
    WHERE video_tags IS NOT NULL AND TRIM(video_tags) <> '';
    '''
    
    mycursor.execute(query)
    data = mycursor.fetchall()
    connection.close()
    
    columns = ["video_id", "channel_id", "video_title", "video_thumbnails", "video_description", 
               "channelTitle", "video_tags", "video_duration", "video_viewCount", "video_likeCount", "video_commentCount"]
    
    return pd.DataFrame(data, columns=columns)

def preprocess_data(video_data):
    video_data['video_tags'] = video_data['video_tags'].fillna('').astype(str).apply(
        lambda x: ','.join(sorted(set(tag.strip().lower() for tag in x.split(','))))
    )
    video_data['video_title'] = video_data['video_title'].fillna('').astype(str)
    video_data['channelTitle'] = video_data['channelTitle'].fillna('').astype(str)
    return video_data

def perform_eda(video_data):
    video_data['num_tags'] = video_data['video_tags'].apply(lambda x: len(x.split(',')))

def vectorize_data(video_data):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tag_vectors = vectorizer.fit_transform(video_data['video_tags'])
    return tag_vectors, vectorizer

def perform_kmeans(tag_vectors, num_clusters=150):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(tag_vectors)
    return kmeans

def save_model(vectorizer, kmeans, video_data):
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    video_data['cluster'] = kmeans.predict(vectorizer.transform(video_data['video_tags']))
    with open('clustered_data.pkl', 'wb') as f:
        pickle.dump(video_data, f)

@st.cache_data
def load_pickles():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('clustered_data.pkl', 'rb') as f:
        video_data = pickle.load(f)
    return vectorizer, kmeans, video_data

video_data = fetch_data_from_mysql()
video_data = preprocess_data(video_data)

perform_eda(video_data)

tag_vectors, vectorizer = vectorize_data(video_data)
kmeans = perform_kmeans(tag_vectors)

save_model(vectorizer, kmeans, video_data)

st.title("YouTube Video Recommendation System")

st.sidebar.title("Video Search & Filter")
search_query = st.sidebar.text_input("Search for a video by title:")

unique_tags = sorted(set(tag for sublist in video_data['video_tags'].str.split(',') for tag in sublist if tag.strip()))
selected_tag = st.sidebar.selectbox("Or select a tag:", options=[''] + unique_tags)

def filter_videos(video_data, search_query, selected_tag):
    if search_query:
        return video_data[video_data['video_title'].str.contains(search_query, case=False)]
    elif selected_tag:
        return video_data[video_data['video_tags'].str.contains(selected_tag, case=False)]
    else:
        return pd.DataFrame()

filtered_videos = filter_videos(video_data, search_query, selected_tag)

if 'selected_video_id' not in st.session_state:
    st.session_state.selected_video_id = None

def play_video(video_id):
    st.session_state.selected_video_id = video_id

if not filtered_videos.empty:
    st.write("Videos matching your search:")
    for index, row in filtered_videos.iterrows():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(row['video_thumbnails'], use_column_width=True)
        with col2:
            st.write(row['video_title'])
            st.write(f"Channel: {row['channelTitle']}")
            st.write(f"Likes: {row.get('video_likeCount', 'N/A')}")
            st.write(f"Views: {row.get('video_viewCount', 'N/A')}")
            st.write(f"Duration: {row['video_duration']}")
            st.write(f"Comments: {row.get('video_commentCount', 'N/A')}")
        
        video_url = f"https://www.youtube.com/watch?v={row['video_id']}"
        if st.button(f"Play {row['video_title']}", key=f"play_{row['video_id']}"):
            play_video(row['video_id'])
            st.video(video_url)
else:
    st.write("No videos found. Try searching by title or selecting a tag.")

st.sidebar.title("Recommendations")

if st.session_state.selected_video_id:
    vectorizer, kmeans, video_data = load_pickles()

    selected_video = video_data[video_data['video_id'] == st.session_state.selected_video_id]
    
    if not selected_video.empty:
        first_video_vector = vectorizer.transform([selected_video.iloc[0]['video_tags']])
        cluster_label = kmeans.predict(first_video_vector)[0]
        recommended_videos = video_data[video_data['cluster'] == cluster_label]
        
        recommended_videos = recommended_videos[recommended_videos['video_id'] != st.session_state.selected_video_id]
        
        if recommended_videos.empty:
            channel = selected_video['channelTitle'].values[0]
            recommended_videos = video_data[video_data['channelTitle'] == channel]
            recommended_videos = recommended_videos[recommended_videos['video_id'] != st.session_state.selected_video_id]
        
        recommended_videos = recommended_videos.drop_duplicates(subset=['video_id']).head(7)

    else:
        recommended_videos = pd.DataFrame()
    
    if not recommended_videos.empty:
        st.sidebar.write("Videos recommended:")
        for index, row in recommended_videos.iterrows():
            st.sidebar.image(row['video_thumbnails'], use_column_width=True)
            st.sidebar.write(row['video_title'])
            st.sidebar.write(f"Channel: {row['channelTitle']}")
            st.sidebar.write(f"Likes: {row.get('video_likeCount', 'N/A')}")
            st.sidebar.write(f"Views: {row.get('video_viewCount', 'N/A')}")
            st.sidebar.write(f"Duration: {row['video_duration']}")
            st.sidebar.write(f"Comments: {row.get('video_commentCount', 'N/A')}")
            video_url = f"https://www.youtube.com/watch?v={row['video_id']}"
            if st.sidebar.button(f"Play {row['video_title']}", key=f"rec_play_{row['video_id']}"):
                play_video(row['video_id'])
                st.video(video_url)
            
              
    st.sidebar.write("Select a video to get recommendations.")

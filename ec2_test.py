import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pickle
import matplotlib.pyplot as plt

rds_host = "streamlit.c7q6aoykitvo.us-east-1.rds.amazonaws.com"
rds_port = "3306"
rds_user = "admin"
rds_password = "admin123"
rds_dbname = "youtube"

def connect_to_mysql():
    try:
        return mysql.connector.connect(
            host=rds_host,
            user=rds_user,
            password=rds_password,
            database=rds_dbname
        )
    except mysql.connector.Error as err:
        st.error(f"Error: {err}")
        return None

def list_tables():
    connection = connect_to_mysql()
    if connection:
        try:
            query = "SHOW TABLES"
            tables = pd.read_sql(query, con=connection)
            return tables
        except Exception as e:
            st.error(f"Error fetching tables: {e}")
        finally:
            connection.close()
    return pd.DataFrame()

def fetch_data_from_mysql(table_name):
    engine = create_engine(f"mysql+pymysql://{rds_user}:{rds_password}@{rds_host}:{rds_port}/{rds_dbname}")
    query = f'''
    SELECT video_id, channel_id, video_title, video_thumbnails, video_description, channelTitle, 
           video_tags, video_duration, video_viewCount, video_likeCount, video_commentCount
    FROM {table_name}
    WHERE video_tags IS NOT NULL AND TRIM(video_tags) <> '';
    '''
    try:
        data = pd.read_sql(query, con=engine)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def preprocess_data(video_data):
    video_data['video_tags'] = video_data['video_tags'].fillna('').astype(str).apply(
        lambda x: ','.join(sorted(set(tag.strip().lower() for tag in x.split(',')))))
    video_data['video_title'] = video_data['video_title'].fillna('').astype(str)
    video_data['channelTitle'] = video_data['channelTitle'].fillna('').astype(str)
    return video_data

def perform_eda(video_data):
    video_data['num_tags'] = video_data['video_tags'].apply(lambda x: len(x.split(',')))

def vectorize_data(video_data):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tag_vectors = vectorizer.fit_transform(video_data['video_tags'])
    return tag_vectors, vectorizer

def perform_kmeans(tag_vectors, num_clusters=200):
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

def calculate_silhouette_score(tag_vectors, kmeans):
    cluster_labels = kmeans.labels_
    silhouette_avg = silhouette_score(tag_vectors, cluster_labels)
    return silhouette_avg

def plot_clusters(tag_vectors, kmeans):
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(tag_vectors.toarray())
    
    cluster_labels = kmeans.labels_
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Cluster Label')
    plt.title('Clusters Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

@st.cache_resource
def load_pickles():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('clustered_data.pkl', 'rb') as f:
        video_data = pickle.load(f)
    return vectorizer, kmeans, video_data

# Streamlit Interface
st.title("YouTube Video Recommendation System")

table_name = "video_data"  
video_data = fetch_data_from_mysql(table_name)
if video_data.empty:
    st.stop() 

video_data = preprocess_data(video_data)
perform_eda(video_data)
tag_vectors, vectorizer = vectorize_data(video_data)
kmeans = perform_kmeans(tag_vectors)
save_model(vectorizer, kmeans, video_data)

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
            st.write(f"Comments: {row.get('video_commentCount', 'N/A')}")
        
        video_url = f"https://www.youtube.com/watch?v={row['video_id']}"
        if st.button(f"Play {row['video_title']}", key=f"play_{row['video_id']}"):
            play_video(row['video_id'])
            st.video(video_url)
else:
    st.write("No videos found. Try searching by title or selecting a tag.")

# Recommendations
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
            st.sidebar.write(f"Comments: {row.get('video_commentCount', 'N/A')}")
else:
    st.sidebar.write("Please select a video to get recommendations.")

# Silhouette Score Calculation
silhouette_avg = calculate_silhouette_score(tag_vectors, kmeans)
st.sidebar.write(f"Silhouette Score: {silhouette_avg}")

# Cluster Visualization
st.write("Cluster Visualization")
plot_clusters(tag_vectors, kmeans)

#pip install pandas scikit-learn numpy surprise
#pip install numpy<2

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from tqdm import tqdm

# Завантаження даних
print("Завантаження даних...")
hotel_details = pd.read_csv("hotel/Hotel_details.csv")
room_attributes = pd.read_csv("hotel/Hotel_Room_attributes.csv")
price_data = pd.read_csv("hotel/hotel_price_min_max - Formula.csv")

# Очищення даних
print("Очищення даних...")
hotel_details.fillna("", inplace=True)
room_attributes.fillna("", inplace=True)
price_data.fillna(0, inplace=True)

# Об’єднання даних
print("Об’єднання даних...")
hotel_data = pd.merge(hotel_details, room_attributes, left_on="hotelid", right_on="hotelcode", how="inner")
hotel_data = pd.merge(hotel_data, price_data, on="hotelcode", how="left")

# Об’єднання текстових ознак
print("Створення текстових ознак...")
hotel_data['combined_features'] = hotel_data['roomamenities'] + " " + hotel_data['roomtype'] + " " + hotel_data['propertytype']

# Використання TF-IDF
print("Побудова TF-IDF матриці...")
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(hotel_data['combined_features'])

# Зменшення розмірності
print("Зменшення розмірності...")
svd = TruncatedSVD(n_components=100)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

# Обчислення косинусної схожості
print("Обчислення косинусної схожості...")
cosine_sim = cosine_similarity(tfidf_reduced, tfidf_reduced)

# Функція для рекомендацій готелів на основі схожості
def get_recommendations(hotel_id, cosine_sim=cosine_sim):
    idx = hotel_data[hotel_data['hotelid'] == hotel_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Вибираємо 5 найсхожіших готелів
    hotel_indices = [i[0] for i in sim_scores]
    return hotel_data.iloc[hotel_indices][['hotelname', 'city', 'country', 'starrating', 'min', 'max']]

# Приклад даних про рейтинги
ratings_dict = {
    "user_id": [1, 2, 3, 4, 5, 1, 2, 3],
    "hotel_id": [101, 102, 103, 104, 105, 102, 103, 104],
    "rating": [5, 3, 4, 2, 5, 4, 3, 5]
}
ratings = pd.DataFrame(ratings_dict)

# Створення датасету для Surprise
print("Підготовка даних для колаборативної фільтрації...")
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[["user_id", "hotel_id", "rating"]], reader)

# Розділення на тренувальний і тестовий набори
print("Розділення даних на тренувальні та тестові...")
trainset, testset = train_test_split(data, test_size=0.2)

# Модель SVD для рекомендацій
print("Навчання моделі SVD...")
svd = SVD()
svd.fit(trainset)

# Оцінка моделі
print("Оцінка моделі...")
predictions = svd.test(testset)

# Функція для рекомендацій готелів для користувача
def recommend_for_user(user_id, n_recommendations=5):
    hotels = hotel_data['hotelid'].unique()
    user_rated_hotels = ratings[ratings['user_id'] == user_id]['hotel_id']
    hotels_to_rate = [hotel for hotel in hotels if hotel not in user_rated_hotels.values]

    # Виведення прогрес-бару
    print(f"Генерація рекомендацій для користувача {user_id}...")
    recommendations = []
    for hotel in tqdm(hotels_to_rate, desc="Processing"):
        pred = svd.predict(user_id, hotel)
        recommendations.append(pred)

    recommendations = sorted(recommendations, key=lambda x: x.est, reverse=True)[:n_recommendations]
    recommended_hotel_ids = [pred.iid for pred in recommendations]
    return hotel_data[hotel_data['hotelid'].isin(recommended_hotel_ids)][['hotelname', 'city', 'starrating', 'min', 'max']]

# Рекомендації для довільного готелю на основі контентно-орієнтованої моделі
print("\nРекомендації для готелю:")
print(get_recommendations(177167))

# Рекомендації для користувача на основі колаборативної фільтрації
print("\nРекомендації для користувача:")
print(recommend_for_user(user_id=1))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# Завантаження даних
print("Завантаження даних...")
hotel_details = pd.read_csv("hotel/Hotel_details.csv")
room_attributes = pd.read_csv("hotel/Hotel_Room_attributes.csv")
price_data = pd.read_csv("hotel/hotel_price_min_max - Formula.csv")

# Очищення даних: заповнення пропусків
print("Очищення даних...")
hotel_details.fillna("", inplace=True)
room_attributes.fillna("", inplace=True)
price_data.fillna(0, inplace=True)

# Об’єднання даних готелів та атрибутів кімнат
print("Об'єднання даних готелів та атрибутів...")
hotel_data = pd.merge(hotel_details, room_attributes, left_on="hotelid", right_on="hotelcode", how="inner")
hotel_data = pd.merge(hotel_data, price_data, on="hotelcode", how="left")

# Токенізація текстових атрибутів, зручностей та описів
print("Токенізація текстових атрибутів...")
hotel_data['combined_features'] = hotel_data['roomamenities'] + " " + hotel_data['roomtype'] + " " + hotel_data['propertytype']

# Використання TF-IDF для перетворення текстових ознак у числові
print("Обчислення TF-IDF...")
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(hotel_data['combined_features'])

# Зменшення розмірності з використанням TruncatedSVD
print("Зменшення розмірності...")
svd = TruncatedSVD(n_components=100, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

# Налаштування NearestNeighbors для знаходження найближчих сусідів
print("Налаштування моделі NearestNeighbors...")
nn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
nn.fit(tfidf_reduced)

# Функція для рекомендацій готелів на основі схожості
def get_recommendations(hotel_id):
    # Перевірка наявності hotel_id в даних
    if hotel_id not in hotel_data['hotelid'].values:
        print(f"Готель з ID {hotel_id} не знайдено.")
        return pd.DataFrame()  # Повертаємо порожній DataFrame, якщо готель не знайдено
    
    # Знаходимо індекс готелю
    idx = hotel_data[hotel_data['hotelid'] == hotel_id].index[0]

    # Отримуємо найближчих сусідів
    distances, indices = nn.kneighbors([tfidf_reduced[idx]])
    similar_indices = indices.flatten()[1:6]  # Вибираємо 5 найближчих готелів

    # Повертаємо інформацію про рекомендовані готелі
    return hotel_data.iloc[similar_indices][['hotelname', 'city', 'country', 'starrating', 'min', 'max']]

# Приклад даних про рейтинги (імітовані)
ratings_dict = {
    "user_id": [1, 2, 3, 4, 5, 1, 2, 3],
    "hotel_id": [101, 102, 103, 104, 105, 102, 103, 104],
    "rating": [5, 3, 4, 2, 5, 4, 3, 5]
}
ratings = pd.DataFrame(ratings_dict)

# Створення dataset для Surprise
print("Підготовка даних для колаборативної фільтрації...")
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[["user_id", "hotel_id", "rating"]], reader)

# Розділення на тренувальний і тестовий набори
trainset, testset = train_test_split(data, test_size=0.2)

# Використання моделі SVD для рекомендацій
print("Навчання моделі SVD...")
svd_model = SVD()
svd_model.fit(trainset)

# Функція рекомендації для користувача на основі колаборативної фільтрації
def recommend_for_user(user_id, n_recommendations=5):
    print(f"Рекомендації для користувача {user_id}...")
    hotels = hotel_data['hotelid'].unique()
    user_rated_hotels = ratings[ratings['user_id'] == user_id]['hotel_id']
    hotels_to_rate = [hotel for hotel in hotels if hotel not in user_rated_hotels.values]

    predictions = [svd_model.predict(user_id, hotel) for hotel in tqdm(hotels_to_rate, desc="Оцінка готелів")]
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:n_recommendations]

    recommended_hotel_ids = [pred.iid for pred in recommendations]
    return hotel_data[hotel_data['hotelid'].isin(recommended_hotel_ids)][['hotelname', 'city', 'starrating', 'min', 'max']]

# Виклик функцій для демонстрації
print("Рекомендації для готелю:")
recommendations_for_hotel = get_recommendations(177167)
if not recommendations_for_hotel.empty:
    print(recommendations_for_hotel)
else:
    print("Рекомендації недоступні для вказаного готелю.")

print("\nРекомендації для користувача:")
recommendations_for_user = recommend_for_user(user_id=1)
print(recommendations_for_user)

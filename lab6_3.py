#pip install pandas scikit-learn numpy surprise

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split

#
hotel_details = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/ktovd/lab 6/hotel/Hotel_details.csv")
room_attributes = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/ktovd/lab 6/hotel/Hotel_Room_attributes.csv")
price_data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/ktovd/lab 6/hotel/hotel_price_min_max - Formula.csv")

# Очищення даних: заповнення пропусків
hotel_details.fillna("", inplace=True)
room_attributes.fillna("", inplace=True)
price_data.fillna(0, inplace=True)

# Об’єднання даних готелів та атрибутів кімнат
hotel_data = pd.merge(hotel_details, room_attributes, left_on="hotelid", right_on="hotelcode", how="inner")

# Об’єднання даних з цінами
hotel_data = pd.merge(hotel_data, price_data, on="hotelcode", how="left")

# Токенізація текстових атрибутів, зручностей та описів
hotel_data['combined_features'] = hotel_data['roomamenities'] + " " + hotel_data['roomtype'] + " " + hotel_data['propertytype']

# Використання TF-IDF для перетворення текстових ознак у числові
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(hotel_data['combined_features'])

# Обчислення косинусної схожості між готелями
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Створення функції для рекомендації готелів
def get_recommendations(hotel_id, cosine_sim=cosine_sim):
    # Знаходимо індекс готелю
    idx = hotel_data[hotel_data['hotelid'] == hotel_id].index[0]

    # Отримуємо список схожості з іншими готелями
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Сортуємо готелі за схожістю
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Вибираємо 5 найсхожіших готелів
    sim_scores = sim_scores[1:6]

    # Отримуємо індекси готелів
    hotel_indices = [i[0] for i in sim_scores]

    # Повертаємо топ-5 рекомендованих готелів
    return hotel_data.iloc[hotel_indices][['hotelname', 'city', 'country', 'starrating', 'min', 'max']]

# Приклад даних про рейтинги (імітовані)
ratings_dict = {
    "user_id": [1, 2, 3, 4, 5, 1, 2, 3],
    "hotel_id": [101, 102, 103, 104, 105, 102, 103, 104],
    "rating": [5, 3, 4, 2, 5, 4, 3, 5]
}
ratings = pd.DataFrame(ratings_dict)

# Створення dataset для Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[["user_id", "hotel_id", "rating"]], reader)

# Розділення на тренувальний і тестовий набори
trainset, testset = train_test_split(data, test_size=0.2)

# Використання моделі SVD для рекомендацій
svd = SVD()
svd.fit(trainset)

# Оцінка моделі на тестовому наборі
predictions = svd.test(testset)


def recommend_for_user(user_id, n_recommendations=5):
    # Пошук всіх готелів, які не були оцінені користувачем
    hotels = hotel_data['hotelid'].unique()
    user_rated_hotels = ratings[ratings['user_id'] == user_id]['hotel_id']
    hotels_to_rate = [hotel for hotel in hotels if hotel not in user_rated_hotels.values]

    # Оцінка та сортування за прогнозованим рейтингом
    predictions = [svd.predict(user_id, hotel) for hotel in hotels_to_rate]
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:n_recommendations]

    # Повертаємо рекомендовані готелі
    recommended_hotel_ids = [pred.iid for pred in recommendations]
    return hotel_data[hotel_data['hotelid'].isin(recommended_hotel_ids)][['hotelname', 'city', 'starrating', 'min', 'max']]

# Рекомендації для довільного готелю на основі контентно-орієнтованої моделі
print(get_recommendations(177167))

# Рекомендації для користувача на основі колаборативної фільтрації
print(recommend_for_user(user_id=1))
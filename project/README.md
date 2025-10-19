# Драйви с Drivee

> Драйви с Drivee - умный помощник, основанный на моделях машинного обучения, учитывает 11 параметров заказа для формирование наивыгодной цены для водителя, на которую с большей вероятностью согласится клиент.

> Разработан с использованием Python и библиотек для ML(PyTorch, Scikit-learn), для анализа данных и создания диаграмм использованы Pandas, Matplotlib. С целью анализа "юзабилити" умного помощника, интегированного в сервис, был разработан интерфейс, с использованием фреймворка Vue, протестирован пользователями на удобство.

>#### Архитектура 

Нейронная сеть использует одно общее тело и две головы (Price Head, Prob Head).
* Общее тело извлекает признаки из входных данных (Параметры входа)
* Price Head предсказывает цену на предстоящую поездку
* Prob Head предсказывает вероятность, с которой пассажир согласится на эту цену

> Запуск проекта

### Бэкенд (Django, Django REST Framework)

python -m venv venv
source venv/bin/activate  # (или venv\Scripts\activate на Windows)
pip install -r requirements.txt

cd project/backend

создаем пустую sql-базу cost_calculator

python manage.py runserver

ENDPOINT: http://127.0.0.1:8000/api/v1/calc/prices/

POST-запрос вида
* {
        "order_timestamp": "2025-01-13 08:15:00",
        "price_start_local": 380,
        "distance_in_meters": 7000,
        "duration_in_seconds": 900,
        "pickup_in_meters": 800,
        "pickup_in_seconds": 120,
        "driver_rating": 4.4,
        "user_rating": 4.95
}

### Фронтенд (Vue Yandex Maps, Vue 3, axios, tailwind css, TypeScript, Pinia)

отдельный терминал

cd project/frontend

npm install

npm run dev


### ML (PyTorch, Pandas, NumPy, Scikit-Learn)

cd project/final_tests

запуск по очереди:

python linear_reg_price_learn.py

python time_model_leaern.py

python pricenet_model_learn.py

python pricenet_v3.py

python final_run_interference.py

последний файл формирует цены в predictions.csv

### АНАЛИТИКА (Pandas, Matplotlab, Seaborn)

cd project/eda_analytics

запускаем eda_analytics.py

получаем аналитику таблицы train.csv


### Структура репозитория

..\project\frontend - фронтенд

..\project\backend - бэкенд

..\project\backend\api\mods, ..\CostCalculator\project\final_tests - ml

..\project\frontend - аналитика

..\project\links - материалы

..\project\data - .csv-файлы

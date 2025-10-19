import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

DATA_FILE = '../../../../Downloads/project/data/train.csv'

sns.set(style='whitegrid', font_scale=1.1)
plt.rcParams['figure.figsize'] = (10, 5)

df = pd.read_csv(DATA_FILE)

# Преобразуем 'done' в 1, 'cancel' в 0
df['is_done_numeric'] = df['is_done'].map({'done': 1, 'cancel': 0})

# преобразование timestamps
df['order_ts'] = pd.to_datetime(df['order_timestamp'])
df['tender_ts'] = pd.to_datetime(df['tender_timestamp'], errors='coerce')
df['driver_reg_date'] = pd.to_datetime(df['driver_reg_date'], errors='coerce')

df['order_hour'] = df['order_ts'].dt.hour
df['order_day_of_week'] = df['order_ts'].dt.dayofweek
df['is_weekend'] = df['order_day_of_week'].isin([5, 6]).astype(int)

# разница начальный цены и бида
df['bid_sum'] = df['price_bid_local'] - df['price_start_local']

# разница биды и нач. цены в процентах
df['bid_percent'] = (df['price_bid_local'] / df['price_start_local'] - 1) * 100

# время работы водителя в сервисе
df['driver_exp'] = (df['order_ts'] - df['driver_reg_date']).dt.days

# Анализ влияния времени суток
plt.figure(figsize=(15, 5))


# 1. Время суток
plt.subplot(1, 3, 1)
hourly_acceptance = df.groupby('order_hour')['is_done_numeric'].mean()
plt.bar(hourly_acceptance.index, hourly_acceptance.values, color='#004b23')
plt.plot(hourly_acceptance.index, hourly_acceptance.values, color='#ccff33', marker='o', markersize=3)
plt.title('Вероятность принятия по часам дня')
plt.xlabel('Час дня')
plt.ylabel('Доля принятых заказов')
plt.grid(True, alpha=0.3)


# 2. День недели
plt.subplot(1, 3, 2)
daily_acceptance = df.groupby('order_day_of_week')['is_done_numeric'].mean()
days = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
plt.bar(range(7), daily_acceptance.values, color='#006400')
plt.title('Вероятность принятия по дням недели')
plt.xlabel('День недели')
plt.ylabel('Доля принятых заказов')
plt.xticks(range(7), days)
plt.grid(True, alpha=0.3)


# 3. Выходные и будни
plt.subplot(1, 3, 3)
weekend_acceptance = df.groupby('is_weekend')['is_done_numeric'].mean()
plt.pie(weekend_acceptance.values, labels=['Будни', 'Выходные'], colors=('#007200', '#008000'), autopct='%1.1f%%')
plt.title('Вероятность принятия: выходные vs будни')
plt.ylabel('Доля принятых заказов')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/time_analytics.png', dpi=300, bbox_inches='tight')
plt.show()


# Анализ влияния цены
plt.figure(figsize=(15, 5))

# 1. Распределение процента надбавки
plt.subplot(1, 3, 1)

# Уберем выбросы для лучшей визуализации
bid_percent_filtered = df['bid_percent'][
    (df['bid_percent'] >= df['bid_percent'].quantile(0.01)) &
    (df['bid_percent'] <= df['bid_percent'].quantile(0.99))
]

plt.hist(bid_percent_filtered, bins=7, color='#38b000')
plt.title('Распределение % надбавки')
plt.xlabel('Процент надбавки')
plt.ylabel('Количество заказов')

# 2. Зависимость вероятности принятия от % надбавки
plt.subplot(1, 3, 2)

# Группируем по округленному проценту надбавки
df['bid_percent_rounded'] = bid_percent_filtered.round()
premium_acceptance = df.groupby('bid_percent_rounded')['is_done_numeric'].mean()
print(premium_acceptance[premium_acceptance >= 0.9].index)
plt.scatter(premium_acceptance.index, premium_acceptance.values, color='#70e000')
plt.title('Зависимость вероятности от % надбавки')
plt.xlabel('Процент надбавки')
plt.ylabel('Доля принятых заказов')
plt.grid(True, alpha=0.3)

# 3. Зависимость принятия от бренда авто
plt.subplot(1, 3, 3)
colors = (
    '#004b23',
    '#006400',
    '#007200',
    '#008000',
    '#38b000',
    '#70e000',
    '#9ef01a',
    '#ccff33',
    )
car_acceptance = df.groupby('carname')['is_done_numeric'].mean()
plt.pie(car_acceptance.values, labels=car_acceptance.index, colors=colors, textprops={'fontsize': 6})
plt.title('Зависимость принятия от бренда авто')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/price_n_brands_analytics.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(15, 5))


# 1. Влияние рейтинга водителя
plt.subplot(1, 3, 1)

rating_bins = pd.cut(df['driver_rating'], bins=np.arange(4.7, 5.05, 0.05))
rating_acceptance = df.groupby(rating_bins)['is_done_numeric'].mean()
rating_acceptance.plot(kind='bar', color='#9ef01a')
plt.title('Влияние рейтинга водителя')
plt.xlabel('Рейтинг водителя')
plt.ylabel('Вероятность')
plt.xticks(rotation=45)

# 2. влияние дистанции
plt.subplot(1, 3, 2)
# Переведем в км и создадим диапазоны
df['distance_km'] = df['distance_in_meters'] / 1000

# Создаем диапазоны по 0.5 км
km_bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, df['distance_km'].max()]
km_labels = ['0-0.5', '0.5-1', '1-1.5', '1.5-2', '2-2.5', '2.5-3', '3-3.5', '3.5-4', '4-4.5', '4.5+']
df['distance_range'] = pd.cut(df['distance_km'], bins=km_bins, labels=km_labels)

distance_acceptance = df.groupby('distance_range')['is_done_numeric'].mean()

plt.bar(distance_acceptance.index, distance_acceptance.values, color='#ccff33')
plt.title('Влияние дистанции поездки')
plt.xlabel('Дистанция (км)')
plt.ylabel('Вероятность')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 3. Влияние времени подъезда
plt.subplot(1, 3, 3)
# Переведем в минуты
df['pickup_minutes'] = df['pickup_in_seconds'] / 60
min_bins = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, df['distance_km'].max()]
min_labels = ['1-1.5', '1.5-2', '2-2.5', '2.5-3', '3-3.5', '3.5-4', '4-4.5', '4.5-5', '5-5.5', '5.5+']
pickup_bins = pd.cut(df['pickup_minutes'], bins=min_bins, labels=min_labels)
pickup_acceptance = df.groupby(pickup_bins)['is_done_numeric'].mean()
pickup_acceptance.plot(kind='line', color='#008000',  marker='o', markersize=3)
plt.title('Влияние времени подъезда')
plt.xlabel('Время подъезда (минуты)')
plt.ylabel('Вероятность')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('../outputs/rt_tm_ds_analytics.png', dpi=300, bbox_inches='tight')
plt.show()

exp_bins = [30, 90, 150, 210, 270, 330, df['driver_exp'].max()]
exp_labels = ['30-90', '90-150', '150-210', '210-270', '270-330', '330+']
exp_bins = pd.cut(df['driver_exp'], bins=exp_bins, labels=exp_labels)
exp_acceptance = df.groupby(exp_bins)['is_done_numeric'].mean()
exp_acceptance.plot(kind='bar', color='#70e000', alpha=0.7)
plt.title('Влияние опыта водителя')
plt.xlabel('Дней с регистрации')
plt.ylabel('Вероятность')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('../outputs/exp_analytics.png', dpi=300, bbox_inches='tight')
plt.show()

# Ключевые выводы
print("\n" + "="*50)
print("КЛЮЧЕВЫЕ ВЫВОДЫ:")
print("="*50)
print(f"Общая доля принятых заказов: {df['is_done_numeric'].mean():.2%}")

best_hour = hourly_acceptance.idxmax()
best_day = daily_acceptance.idxmax()
print(f"Лучший час для заказов: {best_hour}:00 ({hourly_acceptance.max():.2%})")
print(f"Лучший день для заказов: {days[best_day]} ({daily_acceptance.max():.2%})")

# Анализ по проценту надбавки
accepted_orders = df[df['is_done_numeric'] == 1]
rejected_orders = df[df['is_done_numeric'] == 0]

print(f"\nСравнение надбавок:")
print(f"Средний % надбавки для принятых: {accepted_orders['bid_percent'].mean():.2f}%")
print(f"Средний % надбавки для отклоненных: {rejected_orders['bid_percent'].mean():.2f}%")
print(f"Медианный % надбавки для принятых: {accepted_orders['bid_percent'].median():.2f}%")
print(f"Медианный % надбавки для отклоненных: {rejected_orders['bid_percent'].median():.2f}%")

# Найдем оптимальный диапазон надбавок
print(f"\nОптимальный диапазон надбавок:")
optimal_range = premium_acceptance[premium_acceptance >= 0.5]  # где вероятность > 50%
if len(optimal_range) > 0:
    print(f"Надбавки с вероятностью >50%: от {optimal_range.index.min():.1f}% до {optimal_range.index.max():.1f}%")
else:
    print("Нет надбавок с вероятностью >50%")
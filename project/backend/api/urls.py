from django.urls import path
from api import views

urlpatterns = [
    path('prices/', views.PriceRecommendationView.as_view(), name='best-price')
]
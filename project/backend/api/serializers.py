from rest_framework import serializers


class PriceRecommendationSerializer(serializers.Serializer):
    order_timestamp = serializers.CharField()
    price_start_local = serializers.FloatField(min_value=1)
    distance_in_meters = serializers.FloatField(min_value=0)
    duration_in_seconds = serializers.FloatField(min_value=0)
    pickup_in_meters = serializers.FloatField(min_value=0)
    pickup_in_seconds = serializers.FloatField(min_value=0)
    driver_rating = serializers.FloatField(min_value=0, max_value=5)


class InfoSerializer(serializers.Serializer):
    processing_time_ms = serializers.FloatField()
    predicted_waiting_time_sec = serializers.FloatField()
    predicted_base_price_rub = serializers.IntegerField()
    price_anomaly = serializers.FloatField()


class RecommendationResponseSerializer(serializers.Serializer):
    price = serializers.IntegerField()
    probability = serializers.FloatField()
    expected_revenue = serializers.FloatField()
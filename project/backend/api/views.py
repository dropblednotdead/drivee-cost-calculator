from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from api.mods.price_net_v3 import predictor
from .serializers import PriceRecommendationSerializer, RecommendationResponseSerializer, InfoSerializer


class PriceRecommendationView(APIView):
    def post(self, request):
        serializer = PriceRecommendationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        order = serializer.validated_data

        try:
            recommendation, info = predictor.get_recommendations(order)

            response_serializer = RecommendationResponseSerializer(recommendation, many=True)
            return Response({
                "recommendation": response_serializer.data,
                "info": info
            })
        except Exception as e:
            return Response(
                {"error": f"Ошибка модели: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
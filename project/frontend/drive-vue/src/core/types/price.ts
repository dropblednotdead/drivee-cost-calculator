export interface BodyPrice {
  price_start_local: number
  distance_in_meters: number
  duration_in_seconds: number
  pickup_in_meters: number
  pickup_in_seconds: number
  driver_rating: number
  order_timestamp: string
  user_rating: number
}

export interface PriceInfo {
  processing_time_ms: number
  predicted_waiting_time_sec: number
  predicted_base_price_rub: number
  price_anomaly: number
}

export interface Price {
  recommendation: PriceRecommendation[]
  info: PriceInfo
}
export interface PriceRecommendation {
  price: number
  probability: number
}

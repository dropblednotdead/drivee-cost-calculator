import { ref, computed } from 'vue'
import { defineStore } from 'pinia'
import type { BodyPrice, Price } from '../types/price'
import { fetchPrice } from '../api/price'

export const useCounterStore = defineStore('counter', () => {
  const price = ref<Price[]>([])

  const activePrice = ref<number>(0)

  const formData = ref<BodyPrice>({
    price_start_local: 200,
    distance_in_meters: 2000,
    duration_in_seconds: 900,
    pickup_in_meters: 800,
    pickup_in_seconds: 120,
    driver_rating: 4.4,
    order_timestamp: '2025-01-13 08:15:00',
    user_rating: 4.95,
  })

  const getPrice = async (body: BodyPrice) => {
    try {
      price.value = await fetchPrice(body)
    } catch (error) {
      console.warn('Нет подключения к интернету попробуйте позже', error)
    }
  }

  return { getPrice, price, activePrice, formData }
})

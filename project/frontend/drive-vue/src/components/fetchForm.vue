<template>
  <form @submit.prevent="validForm" class="flex flex-col">
    <UIInput v-model="counter.formData.price_start_local" />
    <UIInput v-model="counter.formData.distance_in_meters" />
    <UIInput v-model="counter.formData.duration_in_seconds" />
    <UIInput v-model="counter.formData.pickup_in_meters" />
    <UIInput v-model="counter.formData.pickup_in_seconds" />
    <UIInput v-model="counter.formData.driver_rating" />
    <UIInput v-model="counter.formData.order_timestamp" />
    <UIInput v-model="counter.formData.user_rating" />

    <div class="flex justify-center">
      <button type="submit" class="bg-[#53B525] text-white py-2 rounded-2xl w-50 mt-15">
        Отправить
      </button>
    </div>
  </form>
</template>

<script setup lang="ts">
import UIInput from '@/UI/UIInput.vue'
import { useCounterStore } from '@/core/stores/counter'
import { useRouter } from 'vue-router'

const counter = useCounterStore()
const router = useRouter()

const validForm = async () => {
  if (
    !counter.formData.price_start_local ||
    !counter.formData.distance_in_meters ||
    !counter.formData.duration_in_seconds ||
    !counter.formData.pickup_in_meters ||
    !counter.formData.pickup_in_seconds ||
    !counter.formData.driver_rating ||
    !counter.formData.order_timestamp ||
    !counter.formData.user_rating
  ) {
    return alert('Заполните все поля')
  } else {
    console.log(JSON.stringify(counter.formData))
    await counter.getPrice(counter.formData)
    router.push('/start')
  }
}
</script>

<style scoped></style>

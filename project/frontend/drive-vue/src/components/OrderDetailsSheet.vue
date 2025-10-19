<template>
  <!-- <div class="bg-[#F9F9F9] absolute w-full min-h-10 bottom-82 pb-6 pt-4 px-3">
    <div class="flex flex-row items-center py-2 gap-2">
      <arrow class="" />
      <div class="">
        <span class="flex">ул. Ленина 35</span>
        <span class="bt opacity-45">ул. Ильинская 7</span>
      </div>
    </div>

    <route-item
      :title="'Срочный заказ'"
      :price="300"
      :from="'ул. Горького 67'"
      :to="'ул. Бекетова 2'"
      :status="'Сбалансированный'"
    />
  </div> -->
  <div
    class="bg-white absolute bottom-0 w-full border-t-2 border-[#00000023] p-3 rounded-se-[64px] min-h-[20vh]"
  >
    <cross
      class="absolute right-10 top-7 cursor-pointer"
      :class="isUrgently || isExpectation || isNoTrading ? 'top-16' : ''"
    />

    <h1 v-if="isUrgently" class="text-3xl font-bold opacity-50">Срочный заказ</h1>
    <h1 v-if="isExpectation" class="text-3xl font-bold opacity-50">Готов ждать</h1>
    <h1 v-if="isNoTrading" class="text-3xl font-bold opacity-50">Не готов торговаться</h1>

    <div class="flex flex-row items-center py-5 gap-2 border-b-1 border-[#D9D9D9]">
      <arrow class="" />
      <div class="leading-none">
        <span class="flex">ул. Рождественская 8Б</span>
        <span class="bt opacity-45">ул. Почаинская 17</span>
      </div>
    </div>

    <div class="w-full flex flex-row py-2 mb-2">
      <div
        class="flex-1 text-center border-e-1 px-2 my-1 border-[#D9D9D9] flex flex-col justify-center items-center"
      >
        <span class="opacity-50 text-sm">Владимир</span>
        <div class="flex justify-center items-center mt-[-2px]">
          <span class="opacity-50 text-sm">{{ counter.formData.user_rating }} </span>
          <star class="ml-1" />
        </div>
        <span class="bg-[#53b52540] rounded-xl px-3 py-1 text-xs mt-1">Лояльный</span>
      </div>

      <div class="flex-1 flex flex-col justify-center items-center">
        <div class="relative">
          <span class="text-5xl opacity-80 font-bold tracking-tight"
            >{{ counter.formData.price_start_local }} ₽</span
          >
          <img
            v-if="isUrgently"
            class="absolute top-[-10%] right-[-40%] w-10"
            src="/stok.png"
            alt=""
          />
        </div>
        <span class="opacity-50 text-[11px] mt-[-6px]"
          >{{ useFormatTime(counter.formData.duration_in_seconds) }} мин,
          {{ useFormatDistance(counter.formData.distance_in_meters) }}км</span
        >
      </div>
    </div>

    <div class="w-full flex flex-row items-center my-3">
      <hr class="flex-1 border-[#B9B9B9] border-0.5" />
      <span class="text-center text-xs px-3 text-[#636366] whitespace-nowrap"
        >Предложите оптимальную цену</span
      >
      <hr class="flex-1 border-[#B9B9B9] border-0.5" />
    </div>

    <div class="flex flex-row w-full py-3 gap-2">
      <div class="flex-1">
        <price-selector
          :items="counter.price"
          :toggle="toggle"
          :active="activePrice"
          :class-name="'flex flex-row gap-1 justify-between'"
        />
      </div>
      <selectable-button class="flex-shrink-0">
        <pencil />
      </selectable-button>
    </div>

    <div class="flex justify-center w-full mt-2">
      <button
        class="flex flex-row justify-center border-1 text-base rounded-3xl w-full px-auto py-3"
        :class="isExpectation || isNoTrading ? 'green-gradient text-white' : 'text-[#53B525]'"
      >
        Принять за {{ counter.activePrice }}₽
        <fire v-if="isExpectation || isNoTrading" class="ms-1" :width="15" :heigth="18" />
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import Cross from './icons/Cross.vue'
import Arrow from './icons/Arrow.vue'
import Star from './icons/Star.vue'
import PriceSelector from './PriceSelector.vue'
import { ref } from 'vue'
import SelectableButton from '@/UI/SelectableButton.vue'
import Pencil from './icons/Pencil.vue'
import Fire from './icons/Fire.vue'
import RouteItem from './RouteItem.vue'
import { useCounterStore } from '@/core/stores/counter'
import { useFormatDistance } from '@/core/composables/useFormatDistanse'
import { useFormatTime } from '@/core/composables/useFormatTime'

const counter = useCounterStore()

const activePrice = ref<number>(2)

const isUrgently = ref<boolean>(false)
const isExpectation = ref<boolean>(false)
const isNoTrading = ref<boolean>(false)

const toggle = (index: number) => {
  activePrice.value = index
}
</script>

<style scoped></style>

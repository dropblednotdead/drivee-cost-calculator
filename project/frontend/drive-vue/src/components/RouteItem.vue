<template>
  <h1 v-if="status" class="text-3xl font-bold opacity-50">{{ title }}</h1>
  <div class="flex flex-row w-full">
    <div class="flex flex-row items-center py-2 gap-2">
      <arrow class="" />
      <div class="">
        <span class="flex">{{ from }}</span>
        <span class="bt opacity-45">{{ to }}</span>
      </div>
    </div>

    <div v-if="price" class="flex flex-col w-auto">
      <h1 class="text-3xl flex flex-row">{{ price }}₽</h1>
      <img class="w-12" src="/stok.png" alt="" />
      <span class="bg-[#ffd42a48] rounded-2xl text-center px-1 py-1">{{ status }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  title?: string
  price: number
  status?: Status
  from: string
  to: string
}

type Status = 'Сбалансированный' | 'Торгующийся'

const props = defineProps<Props>()
</script>

<style scoped></style>

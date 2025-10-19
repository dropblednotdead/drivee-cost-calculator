<template>
  <div
    @click="handleClick"
    class="px-3 py-3 min-w-20 rounded-4xl text-[#636366] cursor-pointer border border-[#636366] text-sm font-normal transition-all duration-500 flex flex-row relative items-center justify-center"
    :class="isActive ? 'green-gradient text-white border-0 pe-4' : 'bg-none'"
  >
    <!-- Градиентный фон -->
    <div
      class="absolute rounded-4xl inset-0 green-gradient opacity-0 transition-opacity duration-900"
      :class="{ 'opacity-100': isActive }"
    ></div>

    <!-- Контент поверх -->
    <span class="relative z-10">{{ value }}</span>
    <slot><span :class="index === 2 ? 'me-1 relative z-10' : 'relative z-10'">₽</span></slot>
    <fire
      :width="11"
      :heigth="14"
      class="absolute right-3  transition-colors duration-500"
      v-if="index === 2"
      :currentColor="isActive && index === 2 ? 'white' : '#636366'"
    />
  </div>
</template>

<script setup lang="ts">
import Fire from '@/components/icons/Fire.vue'
import { useCounterStore } from '@/core/stores/counter'
import { computed, onMounted } from 'vue'

interface Props {
  value?: string | number
  isActive?: boolean
  index?: number
}
const props = defineProps<Props>()
const emit = defineEmits(['select'])
const counter = useCounterStore()

onMounted(() => {
  if (props.isActive) {
    counter.activePrice = Number(props.value)
  }
})

const handleClick = () => {
  if (isNaN(Number(props.value))) {
    return
  } else {
    counter.activePrice = Number(props.value)
  }
  emit('select')
}
</script>

<style scoped></style>

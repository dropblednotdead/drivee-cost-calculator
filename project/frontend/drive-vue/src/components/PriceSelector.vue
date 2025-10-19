<template>
  <div :class="className">
    <SelectableButton
      @select="toggle(index)"
      v-for="(value, index) in items.recommendation"
      :key="index"
      :value="value.price"
      :index="index"
      :isActive="active === index"
      :indexAcitive="active"
    />
  </div>
</template>

<script setup lang="ts">
import type { Price } from '@/core/types/price'
import SelectableButton from '@/UI/SelectableButton.vue'

interface Props {
  items: Price
  active: number | boolean
  toggle: (index: number) => void
  className: string
}
const props = defineProps<Props>()
</script>

<style scoped></style>

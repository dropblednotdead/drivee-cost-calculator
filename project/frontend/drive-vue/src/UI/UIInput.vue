<template>
  <input :type="type" class="border-b-1 mt-2 border-[#53B525] bg-[#53b5254d] outline-none px-5 py-2" v-model="value" :placeholder="placeholder" />
</template>

<script setup lang="ts">
interface Props {
  placeholder?: string
  type?: string
}
const props = withDefaults(defineProps<Props>(), {
  type: 'text'
})

const value = defineModel()
</script>

<style scoped></style>

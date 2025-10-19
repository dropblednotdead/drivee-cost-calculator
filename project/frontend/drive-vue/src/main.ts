import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './App.vue'
import router from './core/router'
import { createYmaps } from 'vue-yandex-maps'

import './core/assets/main.css'

const app = createApp(App)

app.use(createPinia())
app.use(router)

app.use(
  createYmaps({
    apikey: '80e48b6a-9fe9-4116-b72b-c97c9ea62d97',
    lang: 'ru_RU',
  }),
)

app.mount('#app')

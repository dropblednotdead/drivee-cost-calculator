import DriverPage from '@/pages/DriverPage.vue'
import FormStart from '@/pages/FormStart.vue'
import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      component: FormStart,
    },
    {
      path: '/start',
      component: DriverPage,
    },
  ],
})

export default router

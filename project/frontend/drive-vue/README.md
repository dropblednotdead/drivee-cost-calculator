# drivee

Этот шаблон поможет тебе начать разработку с **Vue 3** и **Vite**.

## Рекомендуемая среда разработки

[VS Code](https://code.visualstudio.com/) + [Vue (Official)](https://marketplace.visualstudio.com/items?itemName=Vue.volar) (и отключи Vetur).

## Рекомендуемая настройка браузера

- Браузеры на базе Chromium (Chrome, Edge, Brave и т.д.):
  - [Vue.js devtools](https://chromewebstore.google.com/detail/vuejs-devtools/nhdogjmejiglipccpnnnanhbledajbpd)
  - [Включи Custom Object Formatter в Chrome DevTools](http://bit.ly/object-formatters)
- Firefox:
  - [Vue.js devtools](https://addons.mozilla.org/en-US/firefox/addon/vue-js-devtools/)
  - [Включи Custom Object Formatter в Firefox DevTools](https://fxdx.dev/firefox-devtools-custom-object-formatters/)

## Поддержка типов для `.vue` в TypeScript

TypeScript по умолчанию не обрабатывает типы для `.vue`-файлов,  
поэтому для проверки типов вместо `tsc` используется `vue-tsc`.  
В редакторе кода нужно установить [Volar](https://marketplace.visualstudio.com/items?itemName=Vue.volar),  
чтобы TypeScript-сервис корректно понимал `.vue`-типы.

## Настройка конфигурации

Смотри [документацию по настройке Vite](https://vite.dev/config/).

## Установка проекта

```sh
npm install
```

## Сборка и горячая перезагрузка для разработки
```sh
npm run dev
```

## Проверка типов, сборка и минификация для продакшена

```sh
npm run build
```

## Проверка кода с помощью [ESLint](https://eslint.org/)

```sh
npm run lint
```

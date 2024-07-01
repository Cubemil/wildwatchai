<template>
  <div id="app">
    <router-view></router-view>
  </div>
</template>

<script>
import Home from '@/views/Home.vue'
import Gamification from '@/views/Gamification.vue'

export default {
  name: 'App',
  components: {
    Home,
    Gamification
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #40513B;
}

body {
  margin: 0;
  padding: 0;
  background-color: #EDF1D6;
}
</style>

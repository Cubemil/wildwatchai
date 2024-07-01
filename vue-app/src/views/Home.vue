<template>
  <NavBar/>
  <Slideshow/>
</template>

<script>
import NavBar from '@/components/NavBar.vue'
import Slideshow from '@/components/Slideshow.vue'

export default {
  name: 'Home',
  components: {
    Slideshow,
    NavBar
  }
}
</script>

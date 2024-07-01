<template>
  <div class="navbar">
    <ul>
      <li><a class="active">Home</a></li>
      <li><a>Gamification</a></li>
      <li><a>Contact</a></li>
      <li style="float:right"><a>About</a></li>
    </ul>
  </div>
</template>

<script>
export default {
  name: 'NavBar'
}
</script>

<style scoped>
.navbar {
  position: relative;
  top: 0;
  left: 0;
  font-weight: bold;
}

ul {
  list-style-type: none;
  margin: 0;
  padding: 10px;
  overflow: hidden;
  background-color: #333;
}

li {
  float: left;
}

li:last-child {
  border-right: none;
}

li a {
  display: block;
  color: white;
  text-align: center;
  padding: 14px 16px;
  text-decoration: none;
}

li a:hover:not(.active) {
  background-color: #111;
  border-radius: 5px;
}

.active {
  background-color: #04AA6D;
  border-radius: 5px;
}
</style>
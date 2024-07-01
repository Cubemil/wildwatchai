<template>
  <div class="navbar">
    <ul>
      <li>
        <router-link to="/" active-class="active-link">
          <a id="home">Home</a>
        </router-link>
      </li>
      <li>
        <router-link to="/gamification" active-class="active-link">
          <a id="gamification">Gamification</a>
        </router-link>
      </li>
    </ul>
  </div>
</template>

<script>
export default {
  name: 'NavBar',
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
  padding: 8px 10px;
  text-decoration: none;
}

li:hover:not(.active-link) {
  background-color: #111;
  border-radius: 5px;
}

.active-link {
  background-color: #04AA6D;
  border-radius: 5px;
}
</style>

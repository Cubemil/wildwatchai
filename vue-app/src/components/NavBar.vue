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
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

ul {
  list-style-type: none;
  margin: 0;
  padding: 10px;
  overflow: hidden;
  background-color: #609966;
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
  background-color: #9DC08B;
  border-radius: 5px;
}

.active-link {
  background-color: #40513B;
  border-radius: 5px;
}
</style>

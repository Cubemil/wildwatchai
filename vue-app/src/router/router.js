import {createRouter, createWebHistory} from 'vue-router';
import Home from '../views/Home.vue';
import Gamification from '../views/Gamification.vue';

const routes = [
    {path: '/', component: Home},
    {path: '/gamification', component: Gamification}
];

const router = createRouter({
    history: createWebHistory(),
    routes,
});

export default router;
<script setup lang="ts">
import { ref } from 'vue'
import AppHeader from './components/Layout/AppHeader.vue'
import AppSidebar from './components/Layout/AppSidebar.vue'
import ChatHistory from './components/Chat/ChatHistory.vue'
import ChatInput from './components/Chat/ChatInput.vue'

const sidebarCollapsed = ref(false)

const toggleSidebar = () => {
  sidebarCollapsed.value = !sidebarCollapsed.value
}
</script>

<template>
  <div class="h-screen flex flex-col">
    <!-- Header -->
    <AppHeader @toggle-sidebar="toggleSidebar" />

    <!-- Main Content -->
    <div class="flex-1 flex overflow-hidden">
      <!-- Sidebar -->
      <AppSidebar
        :collapsed="sidebarCollapsed"
        class="hidden lg:block"
      />

      <!-- Chat Area -->
      <div class="flex-1 flex flex-col">
        <!-- Messages -->
        <ChatHistory class="flex-1 overflow-y-auto" />

        <!-- Input -->
        <ChatInput />
      </div>
    </div>
  </div>
</template>

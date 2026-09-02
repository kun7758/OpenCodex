<script setup lang="ts">
import { ElButton, ElTag, ElTooltip } from 'element-plus'
import { useChatStore } from '@/stores/chat'

const chatStore = useChatStore()

const emit = defineEmits<{
  toggleSidebar: []
}>()
</script>

<template>
  <header class="h-14 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 flex items-center px-4">
    <!-- 左侧 -->
    <div class="flex items-center gap-3">
      <ElButton
        text
        @click="emit('toggleSidebar')"
        class="lg:hidden"
      >
        <span class="text-xl">&#9776;</span>
      </ElButton>

      <div class="flex items-center gap-2">
        <span class="text-xl">&#129302;</span>
        <h1 class="text-lg font-semibold">OpenNova</h1>
        <span class="text-xs text-gray-500">v0.4.3</span>
      </div>
    </div>

    <!-- 中间 -->
    <div class="flex-1 flex justify-center">
      <ElTag
        :type="chatStore.connected ? 'success' : 'danger'"
        size="small"
        effect="light"
      >
        {{ chatStore.connected ? '已连接' : '未连接' }}
      </ElTag>
      <ElTag
        v-if="chatStore.sessionId"
        type="info"
        size="small"
        effect="plain"
        class="ml-2"
      >
        会话: {{ chatStore.sessionId.substring(0, 8) }}...
      </ElTag>
    </div>

    <!-- 右侧 -->
    <div class="flex items-center gap-2">
      <ElTooltip content="清除对话" placement="bottom">
        <ElButton
          text
          @click="chatStore.clearEvents()"
          :disabled="chatStore.events.length === 0"
        >
          <span>&#128465;</span>
        </ElButton>
      </ElTooltip>

      <!-- <ElTooltip content="GitHub" placement="bottom">
        <ElButton
          text
          tag="a"
          href="https://github.com/Wardell-Stephen-CurryII/OpenNova"
          target="_blank"
        >
          <span>&#128279;</span>
        </ElButton>
      </ElTooltip> -->
    </div>
  </header>
</template>

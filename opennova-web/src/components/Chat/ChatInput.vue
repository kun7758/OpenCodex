<script setup lang="ts">
import { ref, computed } from 'vue'
import { ElInput, ElButton, ElTooltip } from 'element-plus'
import { useChatStore } from '@/stores/chat'
import { useWebSocket } from '@/composables/useWebSocket'

const chatStore = useChatStore()
const { sendMessage, cancelRun } = useWebSocket()

const inputMessage = ref('')
const mode = ref<'act' | 'plan'>('act')

const isDisabled = computed(() => {
  return !chatStore.connected || chatStore.isGenerating || !inputMessage.value.trim()
})

const handleSubmit = () => {
  if (isDisabled.value) return

  const message = inputMessage.value.trim()
  if (message) {
    // 先添加用户消息到事件列表
    chatStore.addEvent({
      type: 'user_message',
      session_id: chatStore.sessionId || '',
      data: { content: message, mode: mode.value },
    })
    // 发送消息到后端
    sendMessage(message, mode.value)
    inputMessage.value = ''
  }
}

const handleKeydown = (e: Event | KeyboardEvent) => {
  if ((e as KeyboardEvent).key === 'Enter' && !(e as KeyboardEvent).shiftKey) {
    e.preventDefault()
    handleSubmit()
  }
}

const handleCancel = () => {
  cancelRun()
}

const toggleMode = () => {
  mode.value = mode.value === 'act' ? 'plan' : 'act'
}
</script>

<template>
  <div class="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
    <!-- 连接状态提示 -->
    <div
      v-if="!chatStore.connected"
      class="mb-3 text-sm text-center py-2 bg-yellow-50 dark:bg-yellow-900/20 text-yellow-600 dark:text-yellow-400 rounded-lg"
    >
      <span class="mr-1">&#9888;</span>
      正在连接服务器...
    </div>

    <!-- 输入区域 -->
    <div class="flex gap-2">
      <!-- 模式切换 -->
      <ElTooltip :content="mode === 'act' ? '当前: 直接执行模式' : '当前: 计划模式'" placement="top">
        <ElButton
          :type="mode === 'plan' ? 'warning' : 'info'"
          @click="toggleMode"
          :disabled="chatStore.isGenerating"
          class="shrink-0"
        >
          {{ mode === 'act' ? '&#9654;' : '&#128203;' }}
        </ElButton>
      </ElTooltip>

      <!-- 输入框 -->
      <ElInput
        v-model="inputMessage"
        type="textarea"
        :rows="2"
        :placeholder="chatStore.isGenerating ? 'AI 正在生成中...' : '输入消息，按 Enter 发送...'"
        :disabled="!chatStore.connected || chatStore.isGenerating"
        @keydown="handleKeydown"
        resize="none"
        class="flex-1"
      />

      <!-- 发送/取消按钮 -->
      <div class="flex flex-col gap-1">
        <ElButton
          v-if="chatStore.isGenerating"
          type="danger"
          @click="handleCancel"
          class="shrink-0"
        >
          取消
        </ElButton>
        <ElButton
          v-else
          type="primary"
          :disabled="isDisabled"
          @click="handleSubmit"
          class="shrink-0"
        >
          发送
        </ElButton>
      </div>
    </div>

    <!-- 提示信息 -->
    <div class="mt-2 text-xs text-gray-400 text-center">
      <span v-if="mode === 'act'">直接执行模式: AI 将直接执行任务</span>
      <span v-else>计划模式: AI 将先生成计划，等待确认后执行</span>
      <span class="mx-2">|</span>
      <span>按 Enter 发送，Shift+Enter 换行</span>
    </div>
  </div>
</template>

<style scoped>
:deep(.el-textarea__inner) {
  background-color: var(--bg-secondary);
  border-color: var(--border-color);
  color: var(--text-primary);
}

:deep(.el-textarea__inner:focus) {
  border-color: var(--accent-color);
}

:deep(.el-textarea__inner::placeholder) {
  color: var(--text-muted);
}
</style>

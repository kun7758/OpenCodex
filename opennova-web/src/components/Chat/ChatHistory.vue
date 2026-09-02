<script setup lang="ts">
import { ref, watch, nextTick, computed } from 'vue'
import { useChatStore } from '@/stores/chat'
import ChatMessage from './ChatMessage.vue'
import type { SDKEvent } from '@/types'

const chatStore = useChatStore()
const messagesContainer = ref<HTMLElement | null>(null)

const scrollToBottom = async () => {
  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

// 将连续的 text_delta 事件合并成一个完整文本
const mergedMessages = computed(() => {
  const result: SDKEvent[] = []
  let currentText = ''

  for (const event of chatStore.events) {
    if (event.type === 'text_delta') {
      // 累积文本内容
      currentText += (event.data.content || '')
    } else {
      // 遇到非 text_delta 事件，先保存累积的文本
      if (currentText) {
        result.push({
          type: 'text_delta',
          session_id: event.session_id,
          data: { content: currentText },
        })
        currentText = ''
      }
      // 添加当前事件
      if (['user_message', 'tool_start', 'tool_result', 'thought', 'plan', 'error'].includes(event.type)) {
        result.push(event)
      }
    }
  }

  // 保存最后累积的文本
  if (currentText) {
    result.push({
      type: 'text_delta',
      session_id: chatStore.sessionId || '',
      data: { content: currentText },
    })
  }

  return result
})

watch(
  () => chatStore.events.length,
  () => {
    scrollToBottom()
  }
)
</script>

<template>
  <div
    ref="messagesContainer"
    class="p-4 space-y-4"
  >
    <!-- 空状态 -->
    <div
      v-if="chatStore.events.length === 0"
      class="flex flex-col items-center justify-center h-full text-gray-400"
    >
      <div class="text-6xl mb-4">&#129302;</div>
      <h2 class="text-xl font-semibold mb-2">OpenNova Web UI</h2>
      <p class="text-sm text-center max-w-md">
        终端 AI 编码助手的 Web 界面。输入消息开始对话，或使用 / 命令执行特殊操作。
      </p>
      <div class="mt-6 grid grid-cols-2 gap-3 text-sm">
        <div class="bg-gray-100 dark:bg-gray-800 rounded-lg p-3 text-center">
          <div class="font-medium mb-1">&#128196; 读取文件</div>
          <div class="text-gray-500">读取 README.md 并说明项目入口</div>
        </div>
        <div class="bg-gray-100 dark:bg-gray-800 rounded-lg p-3 text-center">
          <div class="font-medium mb-1">&#128221; 生成代码</div>
          <div class="text-gray-500">帮我实现文件上传功能</div>
        </div>
        <div class="bg-gray-100 dark:bg-gray-800 rounded-lg p-3 text-center">
          <div class="font-medium mb-1">&#128269; 搜索代码</div>
          <div class="text-gray-500">查找所有 TODO 注释</div>
        </div>
        <div class="bg-gray-100 dark:bg-gray-800 rounded-lg p-3 text-center">
          <div class="font-medium mb-1">&#128295; 执行命令</div>
          <div class="text-gray-500">运行测试并查看结果</div>
        </div>
      </div>
    </div>

    <!-- 消息列表 -->
    <template v-else>
      <template v-for="event in mergedMessages" :key="event.type + JSON.stringify(event.data)">
        <ChatMessage
          :event="event"
          @approve="chatStore.approvePlan"
          @reject="chatStore.rejectPlan"
        />
      </template>

      <!-- 生成中指示器 -->
      <div
        v-if="chatStore.isGenerating"
        class="flex items-center gap-2 text-gray-500 text-sm"
      >
        <div class="flex space-x-1">
          <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0ms"></div>
          <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 150ms"></div>
          <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 300ms"></div>
        </div>
        <span>思考中...</span>
      </div>

      <!-- 错误提示 -->
      <div
        v-if="chatStore.error"
        class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3"
      >
        <div class="flex items-center gap-2 text-sm text-red-600 dark:text-red-400">
          <span>&#9888;</span>
          <span>{{ chatStore.error }}</span>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
@keyframes bounce {
  0%, 100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-4px);
  }
}

.animate-bounce {
  animation: bounce 1s infinite;
}
</style>

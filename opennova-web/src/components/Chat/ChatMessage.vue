<script setup lang="ts">
import { computed } from 'vue'
import { ElTag, ElButton, ElCollapse, ElCollapseItem } from 'element-plus'
import MarkdownRenderer from '../Markdown/MarkdownRenderer.vue'
import type { SDKEvent } from '@/types'

interface Props {
  event: SDKEvent
}

const _props = defineProps<Props>()

const emit = defineEmits<{
  approve: [plan: any]
  reject: []
}>()

const toolStatusText = computed(() => {
  if (_props.event.type === 'tool_start') return '执行中'
  return _props.event.data.success ? '完成' : '失败'
})
</script>

<template>
  <div class="chat-message mb-4">
    <!-- 用户消息 -->
    <div v-if="event.type === 'user_message'" class="flex justify-end">
      <div class="bg-blue-500 text-white rounded-lg px-4 py-2 max-w-[80%]">
        <div>{{ event.data.content }}</div>
      </div>
    </div>

    <!-- AI 文本内容 -->
    <div v-else-if="event.type === 'text_delta'" class="prose max-w-none">
      <MarkdownRenderer :content="event.data.content" />
    </div>

    <!-- 工具开始 -->
    <div
      v-else-if="event.type === 'tool_start'"
      class="bg-gray-50 dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-700"
    >
      <div class="flex items-center gap-2 text-sm">
        <span class="animate-spin text-blue-500">&#9881;</span>
        <span class="text-gray-600 dark:text-gray-300">执行中:</span>
        <ElTag size="small" type="info">{{ event.data.tool_name }}</ElTag>
      </div>
      <div
        v-if="event.data.arguments"
        class="mt-2 text-xs text-gray-500 font-mono bg-gray-100 dark:bg-gray-900 p-2 rounded overflow-x-auto"
      >
        {{ JSON.stringify(event.data.arguments, null, 2) }}
      </div>
    </div>

    <!-- 工具结果 -->
    <div
      v-else-if="event.type === 'tool_result'"
      class="rounded-lg p-3 border"
      :class="{
        'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800': event.data.success,
        'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800': !event.data.success,
      }"
    >
      <div class="flex items-center gap-2 text-sm">
        <span :class="event.data.success ? 'text-green-500' : 'text-red-500'">
          {{ event.data.success ? '&#10003;' : '&#10007;' }}
        </span>
        <span class="text-gray-600 dark:text-gray-300">{{ event.data.tool_name }}</span>
        <ElTag size="small" :type="event.data.success ? 'success' : 'danger'">
          {{ toolStatusText }}
        </ElTag>
      </div>
      <div
        v-if="event.data.output"
        class="mt-2 text-xs font-mono bg-white dark:bg-gray-900 p-2 rounded overflow-x-auto max-h-40 overflow-y-auto"
      >
        {{ event.data.output }}
      </div>
      <div
        v-if="event.data.error"
        class="mt-2 text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/30 p-2 rounded"
      >
        {{ event.data.error }}
      </div>
    </div>

    <!-- 思考过程 -->
    <div
      v-else-if="event.type === 'thought'"
      class="italic text-gray-500 dark:text-gray-400 text-sm pl-4 border-l-2 border-gray-300 dark:border-gray-600"
    >
      <span class="mr-1">&#128173;</span>
      {{ event.data.content }}
    </div>

    <!-- 计划 -->
    <div
      v-else-if="event.type === 'plan'"
      class="border border-blue-200 dark:border-blue-800 rounded-lg p-4 bg-blue-50 dark:bg-blue-900/20"
    >
      <h3 class="font-semibold mb-3 text-blue-800 dark:text-blue-200">
        <span class="mr-2">&#128203;</span>执行计划
      </h3>

      <div v-if="event.data.plan" class="space-y-2">
        <p class="text-sm text-gray-700 dark:text-gray-300">
          <strong>任务:</strong> {{ event.data.plan.task }}
        </p>

        <ElCollapse v-if="event.data.plan.steps?.length">
          <ElCollapseItem
            v-for="step in event.data.plan.steps"
            :key="step.id"
            :title="`${step.id}: ${step.description}`"
          >
            <div class="text-xs text-gray-600 dark:text-gray-400">
              <p v-if="step.tool_hint">工具提示: {{ step.tool_hint }}</p>
              <p>状态: {{ step.status }}</p>
            </div>
          </ElCollapseItem>
        </ElCollapse>

        <div class="mt-4 flex gap-2">
          <ElButton type="primary" size="small" @click="emit('approve', event.data.plan)">
            执行计划
          </ElButton>
          <ElButton size="small" @click="emit('reject')">
            拒绝
          </ElButton>
        </div>
      </div>
    </div>

    <!-- 错误 -->
    <div
      v-else-if="event.type === 'error'"
      class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3"
    >
      <div class="flex items-center gap-2 text-sm text-red-600 dark:text-red-400">
        <span>&#9888;</span>
        <span>{{ event.data.error || '未知错误' }}</span>
      </div>
    </div>
  </div>
</template>

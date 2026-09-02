<script setup lang="ts">
import { ElScrollbar } from 'element-plus'

interface Props {
  collapsed?: boolean
}

defineProps<Props>()

const features = [
  { icon: '&#128196;', name: '文件操作', desc: '读取、写入、编辑文件' },
  { icon: '&#128269;', name: '代码搜索', desc: '全局搜索和正则匹配' },
  { icon: '&#128295;', name: '命令执行', desc: '运行 Shell 命令' },
  { icon: '&#128221;', name: '代码生成', desc: 'AI 生成和修改代码' },
  { icon: '&#128203;', name: '计划模式', desc: '生成执行计划' },
  { icon: '&#128173;', name: '智能思考', desc: 'AI 推理和分析' },
]

const commands = [
  { cmd: '/act', desc: '直接执行任务' },
  { cmd: '/plan', desc: '生成计划' },
  { cmd: '/tools', desc: '查看工具列表' },
  { cmd: '/skills', desc: '查看技能列表' },
  { cmd: '/sessions', desc: '查看会话列表' },
  { cmd: '/help', desc: '查看帮助' },
]
</script>

<template>
  <aside
    class="border-r border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 transition-all duration-300"
    :class="collapsed ? 'w-0 overflow-hidden' : 'w-72'"
  >
    <ElScrollbar>
      <div class="p-4 space-y-6">
        <!-- 功能特性 -->
        <div>
          <h3 class="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
            功能特性
          </h3>
          <div class="space-y-2">
            <div
              v-for="feature in features"
              :key="feature.name"
              class="flex items-start gap-3 p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            >
              <span class="text-lg mt-0.5" v-html="feature.icon"></span>
              <div>
                <div class="text-sm font-medium">{{ feature.name }}</div>
                <div class="text-xs text-gray-500">{{ feature.desc }}</div>
              </div>
            </div>
          </div>
        </div>

        <!-- 快捷命令 -->
        <div>
          <h3 class="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
            快捷命令
          </h3>
          <div class="space-y-1">
            <div
              v-for="command in commands"
              :key="command.cmd"
              class="flex items-center justify-between p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            >
              <code class="text-xs bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded">
                {{ command.cmd }}
              </code>
              <span class="text-xs text-gray-500">{{ command.desc }}</span>
            </div>
          </div>
        </div>

        <!-- 使用提示 -->
        <div>
          <h3 class="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
            使用提示
          </h3>
          <div class="text-xs text-gray-600 dark:text-gray-400 space-y-2">
            <p>&#8226; 每个浏览器标签页拥有独立会话</p>
            <p>&#8226; 会话空闲 1 小时后自动清理</p>
            <p>&#8226; 支持 Markdown 和代码高亮</p>
            <p>&#8226; 可在计划模式下审批执行计划</p>
          </div>
        </div>
      </div>
    </ElScrollbar>
  </aside>
</template>

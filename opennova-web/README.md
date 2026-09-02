# OpenNova Web UI

OpenNova Web UI 是 OpenNova 终端 AI 编码助手的 Web 界面，基于 Vue 3 + Element Plus 构建。

## 功能特性

- 流式输出：实时显示 AI 响应
- Markdown 渲染：支持代码高亮、表格、数学公式
- 工具状态展示：可视化工具执行过程
- 计划模式：支持计划生成和审批
- 多会话支持：每个标签页独立会话

## 技术栈

- Vue 3 + TypeScript
- Element Plus
- Pinia 状态管理
- Tailwind CSS
- markdown-it + highlight.js

## 快速开始

### 前置条件

- Node.js 18+
- npm 或 pnpm

### 安装依赖

```bash
cd opennova-web
npm install
```

### 开发模式

```bash
npm run dev
```

访问 `http://localhost:5173`

### 构建生产版本

```bash
npm run build
```

构建产物输出到 `../src/opennova/static/`

### 预览构建结果

```bash
npm run preview
```

## 项目结构

```
opennova-web/
├── public/                    # 静态资源
├── src/
│   ├── components/           # 组件
│   │   ├── Chat/            # 聊天相关组件
│   │   ├── Markdown/        # Markdown 渲染
│   │   ├── Tools/           # 工具展示
│   │   └── Layout/          # 布局组件
│   ├── composables/         # 组合式函数
│   │   └── useWebSocket.ts  # WebSocket 连接
│   ├── stores/              # Pinia 状态
│   │   └── chat.ts          # 聊天状态管理
│   ├── types/               # TypeScript 类型
│   │   └── index.ts
│   ├── App.vue              # 根组件
│   ├── main.ts              # 入口文件
│   └── style.css            # 全局样式
├── index.html               # HTML 入口
├── vite.config.ts           # Vite 配置
├── tailwind.config.js       # Tailwind 配置
├── tsconfig.json            # TypeScript 配置
└── package.json
```

## 开发指南

### 添加新组件

1. 在 `src/components/` 下创建 `.vue` 文件
2. 使用 `<script setup lang="ts">` 语法
3. 遵循 Vue 3 Composition API 风格

### 状态管理

使用 Pinia 进行状态管理，store 定义在 `src/stores/` 目录。

```typescript
// src/stores/example.ts
import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useExampleStore = defineStore('example', () => {
  const count = ref(0)

  const increment = () => {
    count.value++
  }

  return { count, increment }
})
```

### WebSocket 通信

使用 `useWebSocket` composable 进行 WebSocket 通信：

```typescript
import { useWebSocket } from '@/composables/useWebSocket'

const { sendMessage, cancelRun, connected } = useWebSocket()
```

### 添加新页面

1. 在 `src/views/` 下创建页面组件
2. 在 `src/router/index.ts` 中添加路由配置

## 配置说明

### Vite 配置

开发模式下，Vite 会代理 API 请求到后端服务：

```typescript
// vite.config.ts
server: {
  proxy: {
    '/api': 'http://127.0.0.1:8000',
    '/ws': 'ws://127.0.0.1:8000'
  }
}
```

### 环境变量

创建 `.env` 文件配置环境变量：

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## 部署

### 方式 1：与后端一起部署

1. 构建前端：`npm run build`
2. 启动后端：`uv run opennova serve --port 8000`
3. 访问：`http://localhost:8000`

### 方式 2：独立部署

1. 构建前端：`npm run build`
2. 使用 Nginx 或其他 Web 服务器托管 `dist/` 目录
3. 配置反向代理到后端 API

Nginx 配置示例：

```nginx
server {
    listen 80;
    server_name opennova.example.com;

    location / {
        root /path/to/opennova-web/dist;
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
    }

    location /ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## 常见问题

### 开发模式下无法连接后端

确保后端服务已启动：

```bash
uv run opennova serve --port 8000
```

### 构建后页面空白

检查 `vite.config.ts` 中的 `base` 配置是否正确。

### WebSocket 连接失败

1. 检查后端服务是否运行
2. 检查防火墙设置
3. 确认 WebSocket URL 正确

## 相关链接

- [OpenNova 主项目](https://github.com/Wardell-Stephen-CurryII/OpenNova)
- [Vue 3 文档](https://vuejs.org/)
- [Element Plus 文档](https://element-plus.org/)
- [Pinia 文档](https://pinia.vuejs.org/)

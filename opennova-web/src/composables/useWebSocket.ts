import { ref, onUnmounted } from 'vue'
import { useChatStore } from '@/stores/chat'
import type { SDKEvent } from '@/types'

export function useWebSocket() {
  const chatStore = useChatStore()
  let ws: WebSocket | null = null
  let reconnectTimer: number | null = null
  const reconnectAttempts = ref(0)
  const maxReconnectAttempts = 5

  const getWebSocketUrl = (): string => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    return `${protocol}//${host}/ws/chat`
  }

  const connect = () => {
    if (ws?.readyState === WebSocket.OPEN) {
      return
    }

    const url = getWebSocketUrl()
    ws = new WebSocket(url)

    ws.onopen = () => {
      console.log('WebSocket connected')
      chatStore.setConnected(true)
      reconnectAttempts.value = 0

      // 创建会话
      send({ type: 'create_session' })
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as SDKEvent

        if (data.type === 'session_created') {
          chatStore.setSessionId(data.session_id)
        }

        chatStore.addEvent(data)
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e)
      }
    }

    ws.onclose = (event) => {
      console.log('WebSocket disconnected:', event.code, event.reason)
      chatStore.setConnected(false)

      // 尝试重连
      if (reconnectAttempts.value < maxReconnectAttempts) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.value), 30000)
        reconnectTimer = window.setTimeout(() => {
          reconnectAttempts.value++
          connect()
        }, delay)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }
  }

  const disconnect = () => {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }

    if (ws) {
      ws.close()
      ws = null
    }

    chatStore.setConnected(false)
  }

  const send = (data: Record<string, any>) => {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data))
    } else {
      console.error('WebSocket is not connected')
    }
  }

  const sendMessage = (message: string, mode: string = 'act') => {
    send({
      type: 'chat',
      message,
      mode,
    })
  }

  const cancelRun = () => {
    send({ type: 'cancel' })
  }

  // 自动连接
  connect()

  // 组件卸载时断开连接
  onUnmounted(() => {
    disconnect()
  })

  return {
    connected: chatStore.connected,
    sessionId: chatStore.sessionId,
    connect,
    disconnect,
    sendMessage,
    cancelRun,
  }
}

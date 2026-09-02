import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { SDKEvent, Plan } from '@/types'

export const useChatStore = defineStore('chat', () => {
  // State
  const connected = ref(false)
  const sessionId = ref<string | null>(null)
  const events = ref<SDKEvent[]>([])
  const isGenerating = ref(false)
  const currentPlan = ref<Plan | null>(null)
  const error = ref<string | null>(null)

  // Computed
  const messages = computed(() => {
    return events.value.filter(e =>
      ['text_delta', 'tool_start', 'tool_result', 'thought', 'plan', 'error'].includes(e.type)
    )
  })

  const latestTextContent = computed(() => {
    const textEvents = events.value.filter(e => e.type === 'text_delta')
    return textEvents.map(e => e.data.content || '').join('')
  })

  const hasActiveSession = computed(() => {
    return sessionId.value !== null
  })

  // Actions
  const addEvent = (event: SDKEvent) => {
    events.value.push(event)

    if (event.type === 'run_start') {
      isGenerating.value = true
      error.value = null
    } else if (['run_complete', 'run_error', 'run_cancelled'].includes(event.type)) {
      isGenerating.value = false
      if (event.type === 'run_error') {
        error.value = event.data.error || 'Unknown error'
      }
    } else if (event.type === 'plan') {
      currentPlan.value = event.data.plan
    }
  }

  const clearEvents = () => {
    events.value = []
    currentPlan.value = null
    error.value = null
  }

  const setConnected = (value: boolean) => {
    connected.value = value
    if (!value) {
      sessionId.value = null
      isGenerating.value = false
    }
  }

  const setSessionId = (id: string | null) => {
    sessionId.value = id
  }

  const setError = (message: string | null) => {
    error.value = message
  }

  const approvePlan = () => {
    if (currentPlan.value) {
      currentPlan.value.status = 'approved'
    }
  }

  const rejectPlan = () => {
    currentPlan.value = null
  }

  return {
    // State
    connected,
    sessionId,
    events,
    isGenerating,
    currentPlan,
    error,

    // Computed
    messages,
    latestTextContent,
    hasActiveSession,

    // Actions
    addEvent,
    clearEvents,
    setConnected,
    setSessionId,
    setError,
    approvePlan,
    rejectPlan,
  }
})

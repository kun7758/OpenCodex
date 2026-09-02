export interface SDKEvent {
  type: string
  session_id: string
  data: Record<string, any>
}

export interface SessionInfo {
  session_id: string
  created_at: number
  last_active: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string
  timestamp: number
  events: SDKEvent[]
}

export interface ToolCall {
  id: string
  name: string
  arguments: Record<string, any>
  status: 'pending' | 'running' | 'completed' | 'failed'
  result?: string
  error?: string
}

export interface PlanStep {
  id: string
  description: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  tool_hint?: string
}

export interface Plan {
  task: string
  steps: PlanStep[]
  status: 'draft' | 'awaiting_approval' | 'approved' | 'executing' | 'completed' | 'failed'
}

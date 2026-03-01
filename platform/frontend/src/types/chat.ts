export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'tool'
  content: string
  toolCalls?: ToolCallInfo[]
  timestamp: string
}

export interface ToolCallInfo {
  tool: string
  args: Record<string, unknown>
  status: 'running' | 'done' | 'error'
  resultSummary?: string
}

export interface AgentChatResponse {
  response: string
  tool_calls: ToolCallInfo[]
  session_id: string
  timestamp: string
}

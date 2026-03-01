import { apiFetch } from './client'
import type { ChatMessage } from '../types/chat'

interface ChatRequest {
  message: string
  symbol?: string
  chat_history?: { role: string; content: string }[]
  session_id?: string
}

interface ChatResponse {
  response: string
  session_id?: string
  tool_calls?: { tool: string; args: Record<string, unknown> }[]
  timestamp?: string
}

export async function sendChatMessage(req: ChatRequest): Promise<ChatResponse> {
  return apiFetch<ChatResponse>('/api/chat', {
    method: 'POST',
    body: JSON.stringify(req),
  })
}

export async function sendAgentMessage(req: ChatRequest): Promise<ChatResponse> {
  return apiFetch<ChatResponse>('/api/agent/chat', {
    method: 'POST',
    body: JSON.stringify(req),
  })
}

export function formatHistory(messages: ChatMessage[]) {
  return messages
    .filter((m) => m.role === 'user' || m.role === 'assistant')
    .slice(-10)
    .map((m) => ({ role: m.role, content: m.content }))
}

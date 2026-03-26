import { useCallback, useMemo, useRef, useState } from 'react'
import katex from 'katex'

export type ThoughtStep = {
  step_number: number
  expression: string
  action: string
  action_index: number
}

type StreamState = {
  steps: ThoughtStep[]
  answer: string
  loading: boolean
  error: string | null
  lastProblem: string
}

const INITIAL_STATE: StreamState = {
  steps: [],
  answer: '',
  loading: false,
  error: null,
  lastProblem: '',
}

function toSafeNumber(value: unknown, fallback: number): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function sanitizeLatex(input: string): string {
  const rendered = katex.renderToString(input || '', {
    throwOnError: false,
    strict: 'ignore',
    output: 'htmlAndMathml',
    trust: false,
  })
  return rendered
}

export function useAletheia(endpoint = import.meta.env.VITE_API_ENDPOINT ?? 'http://localhost:8000/v1/solve') {
  const [state, setState] = useState<StreamState>(INITIAL_STATE)
  const controllerRef = useRef<AbortController | null>(null)
  const endpointRef = useRef(endpoint)

  const solve = useCallback(async (problem: string) => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller

    setState({
      steps: [],
      answer: '',
      loading: true,
      error: null,
      lastProblem: problem,
    })

    try {
      const response = await fetch(endpointRef.current, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify({ problem }),
        signal: controller.signal,
      })

      if (!response.ok || !response.body) {
        throw new Error('Brain is Sleeping: Wake up the GPU')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffered = ''
      let currentEvent = ''
      let currentData = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          break
        }

        buffered += decoder.decode(value, { stream: true })
        const lines = buffered.split('\n')
        buffered = lines.pop() ?? ''

        for (const line of lines) {
          const trimmed = line.trimEnd()

          if (trimmed.startsWith('event:')) {
            currentEvent = trimmed.slice(6).trim()
          } else if (trimmed.startsWith('data:')) {
            currentData += `${trimmed.slice(5).trim()}\n`
          } else if (trimmed === '') {
            const payload = currentData.trim()
            if (payload) {
              const parsed = JSON.parse(payload) as Record<string, unknown>
              if (currentEvent === 'step') {
                setState((prev) => ({
                  ...prev,
                  steps: [
                    ...prev.steps,
                    {
                      step_number: toSafeNumber(parsed.step_number, prev.steps.length + 1),
                      expression: sanitizeLatex(String(parsed.expression ?? '')),
                      action: String(parsed.action ?? 'Reason'),
                      action_index: toSafeNumber(parsed.action_index, -1),
                    },
                  ],
                }))
              }

              if (currentEvent === 'answer') {
                setState((prev) => ({
                  ...prev,
                  answer: sanitizeLatex(String(parsed.answer ?? '')),
                  loading: false,
                }))
              }

              if (currentEvent === 'error') {
                throw new Error(String(parsed.detail ?? 'Brain is Sleeping: Wake up the GPU'))
              }
            }
            currentEvent = ''
            currentData = ''
          }
        }
      }

      setState((prev) => ({ ...prev, loading: false }))
    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        return
      }
      setState((prev) => ({
        ...prev,
        loading: false,
        error: 'Brain is Sleeping: Wake up the GPU',
      }))
    }
  }, [])

  return useMemo(() => ({
    ...state,
    solve,
  }), [solve, state])
}

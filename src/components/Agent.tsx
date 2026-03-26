import { FormEvent, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { useAletheia } from '../hooks/useAletheia'

function StepCard({ action, expression }: { action: string; expression: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      className="step-card"
    >
      <div className="step-header">
        <span className="step-action">[{action}]</span>
        <span className="step-check">✓</span>
      </div>
      <div className="katex-line" dangerouslySetInnerHTML={{ __html: expression }} />
    </motion.div>
  )
}

export function Agent() {
  const [problem, setProblem] = useState('')
  const { steps, answer, loading, error, lastProblem, solve } = useAletheia()

  const onSubmit = (event: FormEvent) => {
    event.preventDefault()
    if (!problem.trim()) {
      return
    }
    void solve(problem.trim())
  }

  return (
    <main className="app-shell">
      <section className="chat-shell">
        <AnimatePresence mode="wait">
          {answer ? (
            <motion.div
              key="answer"
              className="final-hero"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
            >
              <div className="katex-hero" dangerouslySetInnerHTML={{ __html: answer }} />
            </motion.div>
          ) : (
            <motion.div
              key="trace"
              className="trace-list"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              {steps.map((step) => (
                <StepCard key={`${step.step_number}-${step.action_index}`} action={step.action} expression={step.expression} />
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {error ? (
          <div className="error-banner">
            <span>{error}</span>
            <button type="button" disabled={!lastProblem} onClick={() => void solve(lastProblem)}>
              Reconnect
            </button>
          </div>
        ) : null}

        <motion.form
          onSubmit={onSubmit}
          className="input-form"
          animate={loading ? { boxShadow: '0 0 0 1px rgba(99, 102, 241, 0.45), 0 0 34px rgba(99, 102, 241, 0.22)' } : { boxShadow: '0 0 0 1px rgba(51, 65, 85, 0.95), 0 0 0 rgba(0, 0, 0, 0)' }}
          transition={{ repeat: loading ? Infinity : 0, duration: 1.2, repeatType: 'mirror' }}
        >
          <input
            value={problem}
            onChange={(e) => setProblem(e.target.value)}
            placeholder="Ask Aletheia to solve a math problem"
            className="chat-input"
            aria-label="math problem"
          />
        </motion.form>
      </section>
    </main>
  )
}

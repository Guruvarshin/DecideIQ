'use client'
import { useState, useEffect, useRef } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { api } from '@/lib/api'

const STEPS = ['Title', 'Upload Docs', 'Your Questions', 'Generate', 'Compare']

function StepBar({ current }) {
  return (
    <div className="flex items-center gap-2 mb-10">
      {STEPS.map((label, i) => (
        <div key={label} className="flex items-center gap-2">
          <div className={`flex items-center gap-1.5 text-sm font-medium ${i < current ? 'text-indigo-600' : i === current ? 'text-gray-900' : 'text-gray-400'}`}>
            <span className={`w-6 h-6 rounded-full text-xs flex items-center justify-center font-bold ${
              i < current ? 'bg-indigo-600 text-white' :
              i === current ? 'bg-indigo-100 text-indigo-700 ring-2 ring-indigo-600' :
              'bg-gray-100 text-gray-400'
            }`}>
              {i < current ? 'v' : i + 1}
            </span>
            <span className="hidden sm:block">{label}</span>
          </div>
          {i < STEPS.length - 1 && (
            <div className={`h-px w-8 ${i < current ? 'bg-indigo-400' : 'bg-gray-200'}`} />
          )}
        </div>
      ))}
    </div>
  )
}

function Card({ children }) {
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-6 animate-fade-in">
      {children}
    </div>
  )
}

// Step 1: title
function StepTitle({ onDone }) {
  const [title, setTitle]   = useState('')
  const [error, setError]   = useState('')
  const [loading, setLoading] = useState(false)

  async function submit(e) {
    e.preventDefault()
    if (!title.trim()) { setError('Enter a comparison title'); return }
    setLoading(true)
    try {
      const { session_id } = await api.createSession(title.trim())
      onDone(session_id)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <h2 className="text-lg font-bold mb-1">What are you comparing?</h2>
      <p className="text-sm text-gray-500 mb-5">
        Be specific, this title drives the question generation.<br />
        <em>e.g. "Job offer comparison", "Health insurance plans", "Smartphone options"</em>
      </p>
      <form onSubmit={submit} className="space-y-4">
        {error && <div className="text-red-600 text-sm">{error}</div>}
        <input
          autoFocus value={title}
          onChange={e => setTitle(e.target.value)}
          placeholder="Job offer comparison"
          className="w-full border border-gray-300 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
        <button
          type="submit" disabled={loading}
          className="w-full bg-indigo-600 text-white font-semibold py-2.5 rounded-lg hover:bg-indigo-700 transition disabled:opacity-50"
        >
          {loading ? 'Creating...' : 'Continue >'}
        </button>
      </form>
    </Card>
  )
}

const UPLOAD_STEPS = [
  { label: 'Parsing document',      duration: 4000  },
  { label: 'Chunking text',         duration: 3000  },
  { label: 'Generating embeddings', duration: 18000 },
  { label: 'Indexing into store',   duration: null  }, // last step — stays active until done
]

function UploadProgress() {
  const [stepIdx, setStepIdx] = useState(0)
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    let idx = 0
    let stepTimer

    const advance = () => {
      if (idx < UPLOAD_STEPS.length - 1) {
        idx++
        setStepIdx(idx)
        const next = UPLOAD_STEPS[idx].duration
        if (next) stepTimer = setTimeout(advance, next)
        // last step has null duration — stays on "Indexing" until upload resolves
      }
    }
    stepTimer = setTimeout(advance, UPLOAD_STEPS[0].duration)

    // Elapsed counter — reassures user the process is alive
    const elapsedTimer = setInterval(() => {
      setElapsed(s => s + 1)
    }, 1000)

    return () => {
      clearTimeout(stepTimer)
      clearInterval(elapsedTimer)
    }
  }, [])

  const currentLabel = UPLOAD_STEPS[stepIdx].label
  const isLast = stepIdx === UPLOAD_STEPS.length - 1

  return (
    <div className="py-2">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin flex-shrink-0" />
          <span className="text-sm text-indigo-600 font-medium">{currentLabel}...</span>
        </div>
        <span className="text-xs text-gray-400 tabular-nums">{elapsed}s</span>
      </div>
      <div className="space-y-1.5">
        {UPLOAD_STEPS.map((s, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className={`w-3 h-3 rounded-full flex-shrink-0 ${
              i < stepIdx  ? 'bg-emerald-500' :
              i === stepIdx ? 'bg-indigo-500 animate-pulse' :
              'bg-gray-200'
            }`} />
            <span className={`text-xs ${
              i < stepIdx  ? 'text-emerald-600 line-through' :
              i === stepIdx ? 'text-indigo-700 font-medium' :
              'text-gray-400'
            }`}>{s.label}</span>
          </div>
        ))}
      </div>
      {isLast && (
        <p className="text-xs text-gray-400 mt-3">
          {elapsed < 30
            ? 'Building the vector index — almost there...'
            : elapsed < 60
            ? 'Large document detected — still working, please wait...'
            : 'Taking longer than usual — this can happen with very large PDFs.'}
        </p>
      )}
    </div>
  )
}

// Step 2: upload docs
function StepUpload({ sessionId, onDone }) {
  const [docs, setDocs]       = useState([])
  const [uploading, setUploading] = useState(false)
  const [error, setError]     = useState('')
  const [docName, setDocName] = useState('')
  const fileRef               = useRef(null)

  async function handleFile(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setError('')
    setUploading(true)
    const fd = new FormData()
    fd.append('file', file)
    if (docName.trim()) fd.append('document_name', docName.trim())
    try {
      const doc = await api.uploadDocument(sessionId, fd)
      setDocs(prev => [...prev, doc])
      setDocName('')
      fileRef.current.value = ''
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  async function handleDelete(idx) {
    try {
      await api.deleteDocument(sessionId, idx)
      setDocs(prev => prev.filter((_, i) => i !== idx))
    } catch (err) {
      setError(err.message)
    }
  }

  const sourceIcon = { pdf: '[PDF]', html: '[HTML]', image: '[IMG]', text: '[TXT]' }

  return (
    <Card>
      <h2 className="text-lg font-bold mb-1">Upload your documents</h2>
      <p className="text-sm text-gray-500 mb-5">
        Upload at least 2 options to compare. Supported: PDF, HTML, PNG/JPG, TXT.
      </p>

      {error && <div className="text-red-600 text-sm mb-3">{error}</div>}

      {docs.length > 0 && (
        <div className="space-y-2 mb-5">
          {docs.map((doc, i) => (
            <div key={i} className="flex items-center justify-between bg-gray-50 border border-gray-200 rounded-lg px-3 py-2">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-indigo-600 bg-indigo-50 px-1.5 py-0.5 rounded">
                  {sourceIcon[doc.source_type] || '[FILE]'}
                </span>
                <div>
                  <div className="text-sm font-medium text-gray-800">{doc.name}</div>
                  <div className="text-xs text-gray-400">{doc.word_count} words - {doc.source_type}</div>
                </div>
              </div>
              <button
                onClick={() => handleDelete(i)}
                className="text-xs text-red-500 hover:text-red-700 transition"
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="space-y-3">
        <input
          value={docName} onChange={e => setDocName(e.target.value)}
          placeholder="Label (optional, e.g. TechCorp Offer)"
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
        <div
          className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-indigo-400 transition"
          onClick={() => fileRef.current?.click()}
        >
          <input ref={fileRef} type="file" className="hidden" onChange={handleFile}
            accept=".pdf,.html,.htm,.txt,.png,.jpg,.jpeg"
          />
          {uploading
            ? <UploadProgress />
            : <p className="text-sm text-gray-500">Click to select a file <span className="text-gray-400">(PDF / HTML / TXT / Image)</span></p>
          }
        </div>
      </div>

      <button
        disabled={docs.length < 2}
        onClick={() => onDone(docs)}
        className="mt-6 w-full bg-indigo-600 text-white font-semibold py-2.5 rounded-lg hover:bg-indigo-700 transition disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {docs.length < 2 ? `Upload ${2 - docs.length} more document${2 - docs.length > 1 ? 's' : ''}` : 'Continue >'}
      </button>
    </Card>
  )
}

// Step 3: user questions
function StepQuestions({ sessionId, onDone }) {
  const [questions, setQuestions] = useState(['', '', ''])
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState('')

  function update(i, v) {
    setQuestions(prev => { const q = [...prev]; q[i] = v; return q })
  }

  async function submit() {
    const filled = questions.map(q => q.trim()).filter(Boolean)
    setLoading(true)
    setError('')
    try {
      if (filled.length > 0) {
        await api.addUserQuestions(sessionId, filled)
      }
      onDone()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <h2 className="text-lg font-bold mb-1">Your questions <span className="text-gray-400 font-normal text-sm">(optional)</span></h2>
      <p className="text-sm text-gray-500 mb-5">
        What specifically matters to you? The AI will rephrase and expand on these.
        Skip if you want fully AI-generated questions.
      </p>

      {error && <div className="text-red-600 text-sm mb-3">{error}</div>}

      <div className="space-y-2 mb-4">
        {questions.map((q, i) => (
          <div key={i} className="flex gap-2">
            <input
              value={q} onChange={e => update(i, e.target.value)}
              placeholder={`Question ${i + 1} - e.g. "What is the base salary?"`}
              className="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
            {questions.length > 1 && (
              <button
                onClick={() => setQuestions(prev => prev.filter((_, j) => j !== i))}
                className="text-gray-400 hover:text-red-500 transition text-lg leading-none"
              >x</button>
            )}
          </div>
        ))}
      </div>

      <button
        onClick={() => setQuestions(prev => [...prev, ''])}
        className="text-sm text-indigo-600 hover:underline mb-6"
      >
        + Add question
      </button>

      <button
        onClick={submit} disabled={loading}
        className="w-full bg-indigo-600 text-white font-semibold py-2.5 rounded-lg hover:bg-indigo-700 transition disabled:opacity-50"
      >
        {loading ? 'Saving...' : 'Continue >'}
      </button>
    </Card>
  )
}

// Step 4: generate + review questions
function StepGenerate({ sessionId, onDone }) {
  const [questions, setQuestions] = useState(null)
  const [loading, setLoading]     = useState(true)
  const [error, setError]         = useState('')

  useEffect(() => {
    api.generateQuestions(sessionId)
      .then(data => setQuestions(data.questions))
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [sessionId])

  return (
    <Card>
      <h2 className="text-lg font-bold mb-1">Generated questions</h2>
      <p className="text-sm text-gray-500 mb-5">
        Rephrased your questions + 5 AI-generated ones. These will be asked to every document.
      </p>

      {loading && (
        <div className="flex items-center gap-3 py-8 justify-center">
          <div className="w-5 h-5 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-gray-500">Generating questions from your title...</span>
        </div>
      )}

      {error && <div className="text-red-600 text-sm">{error}</div>}

      {questions && (
        <>
          <ol className="space-y-2 mb-6">
            {questions.map((q, i) => (
              <li key={i} className="flex gap-2 text-sm">
                <span className="flex-shrink-0 w-5 h-5 rounded-full bg-indigo-100 text-indigo-700 text-xs font-bold flex items-center justify-center">
                  {i + 1}
                </span>
                <span className="text-gray-800">{q}</span>
              </li>
            ))}
          </ol>
          <button
            onClick={onDone}
            className="w-full bg-indigo-600 text-white font-semibold py-2.5 rounded-lg hover:bg-indigo-700 transition"
          >
            Run Comparison
          </button>
        </>
      )}
    </Card>
  )
}

// Comparison progress — timer-based stages that reflect the real pipeline
// Pipeline per question (sequential per doc): retrieval + answer + scoring
// Final step: verdict generation
function CompareProgress({ nQuestions, nDocs }) {
  const PER_Q_MS   = 9000   // ~9s per question (retrieval + answer + score per doc sequentially)
  const VERDICT_MS = 10000  // ~10s for Claude Sonnet verdict

  const totalMs = nQuestions * PER_Q_MS + VERDICT_MS

  const stages = [
    ...Array.from({ length: nQuestions }, (_, i) => ({
      label: `Question ${i + 1} of ${nQuestions}`,
      sub:   'Retrieving context, answering, scoring across all documents',
      ms:    PER_Q_MS,
    })),
    { label: 'Writing verdict', sub: 'Claude Sonnet synthesising final recommendation', ms: VERDICT_MS },
  ]

  const [stageIdx, setStageIdx] = useState(0)
  const [elapsed,  setElapsed]  = useState(0)
  const [qDone,    setQDone]    = useState(0)

  useEffect(() => {
    let idx = 0
    let timer

    const advance = () => {
      idx++
      if (idx < stages.length) {
        setStageIdx(idx)
        if (idx < nQuestions) setQDone(idx)
        timer = setTimeout(advance, stages[idx].ms)
      }
    }
    timer = setTimeout(advance, stages[0].ms)

    const tick = setInterval(() => setElapsed(s => s + 1), 1000)
    return () => { clearTimeout(timer); clearInterval(tick) }
  }, [])

  const isVerdict   = stageIdx === stages.length - 1
  const pct         = Math.min(Math.round((elapsed / (totalMs / 1000)) * 100), 95)
  const currentStage = stages[stageIdx]

  return (
    <div className="py-2">
      {/* Top: current stage */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin flex-shrink-0" />
          <span className="text-sm font-semibold text-indigo-700">{currentStage.label}</span>
        </div>
        <span className="text-xs text-gray-400 tabular-nums">{elapsed}s</span>
      </div>

      <p className="text-xs text-gray-500 mb-4">{currentStage.sub}</p>

      {/* Progress bar */}
      <div className="w-full bg-gray-100 rounded-full h-2 mb-4">
        <div
          className="h-2 rounded-full bg-indigo-500 transition-all duration-1000"
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Question tracker */}
      {!isVerdict && (
        <div className="flex flex-wrap gap-1.5 mb-4">
          {Array.from({ length: nQuestions }, (_, i) => (
            <span key={i} className={`w-6 h-6 rounded-full text-xs flex items-center justify-center font-bold ${
              i < qDone   ? 'bg-emerald-500 text-white' :
              i === qDone ? 'bg-indigo-500 text-white animate-pulse' :
              'bg-gray-200 text-gray-400'
            }`}>
              {i + 1}
            </span>
          ))}
        </div>
      )}

      {/* Stage list */}
      <div className="space-y-1.5">
        {[
          { label: `Answering ${nQuestions} questions`, done: isVerdict, active: !isVerdict },
          { label: 'Writing verdict',                   done: false,      active: isVerdict  },
        ].map((s, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${
              s.done   ? 'bg-emerald-500' :
              s.active ? 'bg-indigo-500 animate-pulse' :
              'bg-gray-200'
            }`} />
            <span className={`text-xs ${
              s.done   ? 'text-emerald-600 line-through' :
              s.active ? 'text-indigo-700 font-medium' :
              'text-gray-400'
            }`}>{s.label}</span>
          </div>
        ))}
      </div>

      <p className="text-xs text-gray-400 mt-4">
        {elapsed < 30  ? 'Processing questions...' :
         elapsed < 90  ? 'Large documents take longer — still working...' :
                         'Almost done — writing your verdict...'}
      </p>
    </div>
  )
}

// Step 5: run comparison
function StepCompare({ sessionId }) {
  const router  = useRouter()
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState('')
  const [nQuestions, setNQuestions] = useState(8)   // default estimate
  const [nDocs,      setNDocs]      = useState(2)

  // Fetch session to get real question + doc count for accurate progress
  useEffect(() => {
    api.getSession(sessionId).then(s => {
      setNDocs(s.documents?.length || 2)
    }).catch(() => {})
    api.getQuestions(sessionId).then(q => {
      setNQuestions(q.generated_questions?.length || 8)
    }).catch(() => {})
  }, [sessionId])

  async function run() {
    setError('')
    setLoading(true)
    try {
      await api.runComparison(sessionId)
      router.push(`/sessions/${sessionId}`)
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  return (
    <Card>
      <h2 className="text-lg font-bold mb-1">Run comparison</h2>
      <p className="text-sm text-gray-500 mb-5">
        The AI retrieves context per document, answers every question, scores comparatively, and writes a verdict.
      </p>

      {error && <div className="text-red-600 text-sm mb-4">{error}</div>}

      {loading ? (
        <CompareProgress nQuestions={nQuestions} nDocs={nDocs} />
      ) : (
        <button
          onClick={run}
          className="w-full bg-emerald-600 text-white font-bold py-3 rounded-lg hover:bg-emerald-700 transition text-base"
        >
          Start Comparison
        </button>
      )}
    </Card>
  )
}

// Main wizard
export default function NewSession() {
  const [step, setStep]           = useState(0)
  const [sessionId, setSessionId] = useState(null)

  return (
    <div className="min-h-screen">
      <nav className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between">
        <Link href="/dashboard" className="flex items-center text-xl">
          <img src="/logo.svg" alt="DecideIQ" className="h-8 w-8 rounded-lg" /><span className="ml-1 font-bold text-gray-900">Decide<span className="text-indigo-600">IQ</span></span>
        </Link>
        <Link href="/dashboard" className="text-sm text-gray-500 hover:text-gray-800">
          &lt; Dashboard
        </Link>
      </nav>

      <div className="max-w-xl mx-auto px-4 py-10">
        <StepBar current={step} />

        {step === 0 && (
          <StepTitle onDone={id => { setSessionId(id); setStep(1) }} />
        )}
        {step === 1 && (
          <StepUpload sessionId={sessionId} onDone={() => setStep(2)} />
        )}
        {step === 2 && (
          <StepQuestions sessionId={sessionId} onDone={() => setStep(3)} />
        )}
        {step === 3 && (
          <StepGenerate sessionId={sessionId} onDone={() => setStep(4)} />
        )}
        {step === 4 && (
          <StepCompare sessionId={sessionId} />
        )}
      </div>
    </div>
  )
}

'use client'
import { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import Link from 'next/link'
import ReactMarkdown from 'react-markdown'
import { api } from '@/lib/api'

// ── Utilities ────────────────────────────────────────────────────────────────
function pct(v) { return Math.round(v * 100) }

function SourceBadge({ source }) {
  const map = {
    rag:                  'bg-blue-100 text-blue-700',
    web:                  'bg-orange-100 text-orange-700',
    full_context:         'bg-gray-100 text-gray-600',
    not_mentioned:        'bg-red-100 text-red-600',
    rag_low_confidence:   'bg-yellow-100 text-yellow-700',
  }
  const labels = {
    rag: 'RAG', web: 'Web', full_context: 'Full doc',
    not_mentioned: 'Not found', rag_low_confidence: 'Low conf.',
  }
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${map[source] || 'bg-gray-100 text-gray-500'}`}>
      {labels[source] || source}
    </span>
  )
}

function ScoreBar({ value, max = 10, color = 'bg-indigo-500' }) {
  const w = Math.round((value / max) * 100)
  return (
    <div className="w-full bg-gray-100 rounded-full h-1.5 mt-1">
      <div className={`h-1.5 rounded-full ${color}`} style={{ width: `${w}%` }} />
    </div>
  )
}

function MetricRow({ label, value }) {
  if (value === undefined || value === null) return null
  return (
    <div>
      <div className="flex justify-between text-xs mb-0.5">
        <span className="text-gray-500">{label}</span>
        <span className="font-semibold text-gray-700">{value.toFixed(3)}</span>
      </div>
      <ScoreBar value={value} max={1} color="bg-indigo-400" />
    </div>
  )
}

// ── Score cards row ──────────────────────────────────────────────────────────
function ScoreCards({ summaries, winner }) {
  return (
    <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${summaries.length}, 1fr)` }}>
      {summaries.map(s => {
        const isWinner = s.doc_name === winner
        return (
          <div key={s.doc_name} className={`rounded-xl border-2 p-5 text-center ${isWinner ? 'border-emerald-400 bg-emerald-50' : 'border-gray-200 bg-white'}`}>
            {isWinner && <div className="text-xs font-bold text-emerald-600 mb-2 uppercase tracking-wide">🏆 Winner</div>}
            <div className="text-2xl font-bold text-gray-900">{s.percentage}%</div>
            <div className="text-sm font-semibold text-gray-700 mt-1 truncate">{s.doc_name}</div>
            <ScoreBar value={s.percentage} max={100} color={isWinner ? 'bg-emerald-500' : 'bg-indigo-400'} />
            <div className="text-xs text-gray-400 mt-2">Raw score: {s.raw_score}</div>
          </div>
        )
      })}
    </div>
  )
}

// ── Q&A breakdown table ──────────────────────────────────────────────────────
function QATable({ questionResults }) {
  const [expanded, setExpanded] = useState(null)

  return (
    <div className="space-y-2">
      {questionResults.map((qr, i) => (
        <div key={i} className="bg-white border border-gray-200 rounded-xl overflow-hidden">
          <button
            onClick={() => setExpanded(expanded === i ? null : i)}
            className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-gray-50 transition"
          >
            <span className="text-sm font-medium text-gray-800 flex-1 pr-4">{qr.question}</span>
            <div className="flex items-center gap-2 flex-shrink-0">
              {qr.per_doc.map(pd => (
                <span key={pd.doc_name} className="text-xs font-bold text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded">
                  {pd.doc_name.split(' ')[0]}: {pd.score}/10
                </span>
              ))}
              <span className="text-gray-400 text-sm">{expanded === i ? '▲' : '▼'}</span>
            </div>
          </button>

          {expanded === i && (
            <div className="border-t border-gray-100 divide-y divide-gray-100">
              {qr.per_doc.map(pd => (
                <div key={pd.doc_name} className="px-4 py-3">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-xs font-bold text-gray-600">{pd.doc_name}</span>
                    <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                      pd.score >= 8 ? 'bg-emerald-100 text-emerald-700' :
                      pd.score >= 5 ? 'bg-yellow-100 text-yellow-700' :
                      'bg-red-100 text-red-600'
                    }`}>
                      {pd.score}/10
                    </span>
                    <SourceBadge source={pd.source} />
                    {pd.grounding_score > 0 && (
                      <span className="text-xs text-gray-400">
                        grounding: {pd.grounding_score.toFixed(2)}
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-700 leading-relaxed">{pd.answer.replace(/—/g, ',')}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

// ── RAGAS eval section ───────────────────────────────────────────────────────
function EvalSection({ sessionId, documents }) {
  const [evals, setEvals]     = useState({})
  const [running, setRunning] = useState({})
  const [error, setError]     = useState('')

  useEffect(() => {
    api.getEvaluation(sessionId).then(setEvals).catch(() => {})
  }, [sessionId])

  async function runEval(docIdx) {
    setError('')
    setRunning(r => ({ ...r, [docIdx]: true }))
    try {
      const result = await api.runEvaluation(sessionId, docIdx)
      setEvals(prev => ({ ...prev, [`doc_${docIdx}`]: result }))
    } catch (err) {
      setError(err.message)
    } finally {
      setRunning(r => ({ ...r, [docIdx]: false }))
    }
  }

  return (
    <div>
      <h2 className="text-xl font-bold text-gray-900 mb-1">RAGAS Evaluation</h2>
      <p className="text-sm text-gray-500 mb-5">
        Measures answer quality using faithfulness and relevancy, no golden answers needed.
      </p>
      {error && <div className="text-red-600 text-sm mb-3">{error}</div>}
      <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${documents.length}, 1fr)` }}>
        {documents.map((doc, i) => {
          const ev = evals[`doc_${i}`]
          return (
            <div key={i} className="bg-white border border-gray-200 rounded-xl p-4">
              <div className="text-sm font-semibold text-gray-700 mb-3 truncate">{doc.name}</div>
              {ev ? (
                <div className="space-y-3">
                  <MetricRow label="Faithfulness" value={ev.faithfulness} />
                  <MetricRow label="Answer Relevancy" value={ev.answer_relevancy} />
                  <div className="pt-2 border-t border-gray-100">
                    <div className="flex justify-between text-sm">
                      <span className="font-semibold text-gray-700">Confidence</span>
                      <span className="font-bold text-indigo-600">{(ev.confidence_score * 100).toFixed(1)}%</span>
                    </div>
                    <ScoreBar value={ev.confidence_score} max={1} color="bg-indigo-500" />
                  </div>
                  <div className="text-xs text-gray-400">{ev.n_questions} questions evaluated</div>
                </div>
              ) : (
                <button
                  onClick={() => runEval(i)}
                  disabled={running[i]}
                  className="w-full text-sm bg-gray-100 hover:bg-indigo-50 hover:text-indigo-700 text-gray-600 font-medium py-2 rounded-lg transition disabled:opacity-50"
                >
                  {running[i] ? (
                    <span className="flex items-center justify-center gap-2">
                      <span className="w-3 h-3 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" />
                      Evaluating…
                    </span>
                  ) : 'Run Evaluation'}
                </button>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Main page ────────────────────────────────────────────────────────────────
export default function SessionResults() {
  const { id } = useParams()
  const [session, setSession]       = useState(null)
  const [comparison, setComparison] = useState(null)
  const [loading, setLoading]       = useState(true)
  const [error, setError]           = useState('')

  useEffect(() => {
    async function load() {
      try {
        const [sess, cmp] = await Promise.all([
          api.getSession(id),
          api.getComparison(id).catch(() => null),
        ])
        setSession(sess)
        setComparison(cmp)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [id])

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-8 h-8 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-600 mb-4">{error}</p>
          <Link href="/dashboard" className="text-indigo-600 hover:underline">← Back to dashboard</Link>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen">
      {/* Nav */}
      <nav className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between sticky top-0 z-10">
        <Link href="/dashboard" className="flex items-center text-xl">
          <img src="/logo.svg" alt="DecideIQ" className="h-8 w-8 rounded-lg" /><span className="ml-1 font-bold text-gray-900">Decide<span className="text-indigo-600">IQ</span></span>
        </Link>
        <Link href="/dashboard" className="text-sm text-gray-500 hover:text-gray-800">← Dashboard</Link>
      </nav>

      <div className="max-w-4xl mx-auto px-4 py-10 space-y-10">

        {/* Header */}
        <div>
          <p className="text-xs font-medium text-gray-400 uppercase tracking-widest mb-1">Comparison</p>
          <h1 className="text-3xl font-bold text-gray-900">{session?.title || 'Untitled'}</h1>
        </div>

        {/* No comparison yet */}
        {!comparison && (
          <div className="bg-white border border-gray-200 rounded-xl p-8 text-center">
            <p className="text-gray-500 mb-4">Comparison not run yet.</p>
            <Link
              href="/sessions/new"
              className="inline-block text-sm text-indigo-600 font-medium hover:underline"
            >
              Go back to complete the setup →
            </Link>
          </div>
        )}

        {comparison && (
          <>
            {/* Winner banner */}
            <div className="bg-emerald-50 border border-emerald-200 rounded-xl px-6 py-4 flex items-center gap-4">
              <span className="text-3xl">🏆</span>
              <div>
                <p className="text-xs font-medium text-emerald-600 uppercase tracking-wide">Winner</p>
                <p className="text-xl font-bold text-emerald-800">{comparison.winner_name}</p>
              </div>
            </div>

            {/* Score cards */}
            <div>
              <h2 className="text-xl font-bold text-gray-900 mb-4">Scores</h2>
              <ScoreCards
                summaries={comparison.doc_summaries}
                winner={comparison.winner_name}
              />
            </div>

            {/* Verdict */}
            <div>
              <h2 className="text-xl font-bold text-gray-900 mb-4">AI Verdict</h2>
              <div className="bg-white border border-gray-200 rounded-xl p-6">
                <div className="prose prose-sm prose-headings:font-bold prose-headings:text-gray-900 prose-p:text-gray-700 prose-p:leading-relaxed prose-strong:text-gray-900 max-w-none">
                  <ReactMarkdown>{comparison.verdict.replace(/—/g, ',')}</ReactMarkdown>
                </div>
              </div>
            </div>

            {/* Q&A breakdown */}
            <div>
              <h2 className="text-xl font-bold text-gray-900 mb-4">
                Question Breakdown
                <span className="ml-2 text-sm font-normal text-gray-400">
                  ({comparison.question_results.length} questions, click to expand)
                </span>
              </h2>
              <QATable questionResults={comparison.question_results} />
            </div>

            {/* RAGAS */}
            {session?.documents && (
              <EvalSection sessionId={id} documents={session.documents} />
            )}
          </>
        )}
      </div>
    </div>
  )
}

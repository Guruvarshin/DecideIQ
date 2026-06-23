'use client'
import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { api } from '@/lib/api'

function statusBadge(status) {
  const map = {
    pending:  'bg-yellow-100 text-yellow-700',
    compared: 'bg-emerald-100 text-emerald-700',
  }
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${map[status] || 'bg-gray-100 text-gray-600'}`}>
      {status}
    </span>
  )
}

export default function Dashboard() {
  const router = useRouter()
  const [sessions, setSessions]   = useState([])
  const [loading, setLoading]     = useState(true)
  const [error, setError]         = useState('')
  const [deleting, setDeleting]   = useState(null)  // session_id being deleted

  async function handleDelete(e, sessionId) {
    e.preventDefault()   // prevent Link navigation
    e.stopPropagation()
    if (!confirm('Delete this comparison? This cannot be undone.')) return
    setDeleting(sessionId)
    try {
      await api.deleteSession(sessionId)
      setSessions(prev => prev.filter(s => s.session_id !== sessionId))
    } catch (err) {
      setError(err.message)
    } finally {
      setDeleting(null)
    }
  }

  useEffect(() => {
    // Ping backend silently to wake Render free tier before user starts uploading.
    // Ignored if it fails — purely for cold start warmup.
    fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/health`).catch(() => {})

    api.listSessions()
      .then(setSessions)
      .catch(err => {
        if (err.message.includes('401') || err.message.toLowerCase().includes('not authenticated')) {
          router.push('/login')
        } else {
          setError(err.message)
        }
      })
      .finally(() => setLoading(false))
  }, [])

  function formatDate(iso) {
    return new Date(iso).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })
  }

  async function handleLogout() {
    await api.logout().catch(() => {})
    router.push('/login')
  }

  return (
    <div className="min-h-screen">
      {/* Nav */}
      <nav className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between">
        <Link href="/dashboard" className="flex items-center text-xl">
          <img src="/logo.svg" alt="DecideIQ" className="h-8 w-8 rounded-lg" /><span className="ml-1 font-bold text-gray-900">Decide<span className="text-indigo-600">IQ</span></span>
        </Link>
        <button onClick={handleLogout} className="text-sm text-gray-500 hover:text-gray-800 transition">
          Sign out
        </button>
      </nav>

      <div className="max-w-4xl mx-auto px-4 py-10">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Your Comparisons</h1>
            <p className="text-gray-500 text-sm mt-1">Each session compares multiple documents side by side</p>
          </div>
          <Link
            href="/sessions/new"
            className="px-4 py-2 bg-indigo-600 text-white font-semibold text-sm rounded-lg hover:bg-indigo-700 transition"
          >
            + New Comparison
          </Link>
        </div>

        {loading && (
          <div className="text-center py-20 text-gray-400">Loading sessions...</div>
        )}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-lg">{error}</div>
        )}

        {!loading && sessions.length === 0 && (
          <div className="text-center py-20">
            <div className="text-5xl mb-4">ðŸ¤”</div>
            <p className="text-gray-500">No comparisons yet. Start your first one.</p>
            <Link href="/sessions/new" className="mt-4 inline-block text-indigo-600 font-medium hover:underline">
              Create comparison â†’
            </Link>
          </div>
        )}

        <div className="space-y-3">
          {sessions.map(s => (
            <div key={s.session_id} className="relative group">
              <Link
                href={`/sessions/${s.session_id}`}
                className="block bg-white border border-gray-200 rounded-xl px-5 py-4 hover:border-indigo-300 hover:shadow-sm transition animate-fade-in"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {statusBadge(s.status)}
                    <span className="font-semibold text-gray-800">
                      {s.title || 'Untitled comparison'}
                    </span>
                  </div>
                  <span className="text-xs text-gray-400">{formatDate(s.created_at)}</span>
                </div>
                <div className="mt-2 flex items-center gap-4 text-xs text-gray-500">
                  <span>{s.document_count} document{s.document_count !== 1 ? 's' : ''}</span>
                  {s.winner_name && (
                    <span className="text-emerald-600 font-medium">Winner: {s.winner_name}</span>
                  )}
                </div>
              </Link>
              <button
                onClick={e => handleDelete(e, s.session_id)}
                disabled={deleting === s.session_id}
                className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity
                           text-xs text-red-400 hover:text-red-600 bg-white border border-red-200
                           hover:border-red-400 px-2 py-1 rounded-md disabled:opacity-40"
              >
                {deleting === s.session_id ? '...' : 'Delete'}
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}


import Link from 'next/link'

export default function Landing() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Nav */}
      <nav className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2">
          <img src="/logo.svg" alt="DecideIQ" className="h-8 w-8 rounded-lg" />
          <span className="font-bold text-gray-900 text-xl">Decide<span className="text-indigo-600">IQ</span></span>
        </Link>
        <div className="flex items-center gap-3">
          <Link href="/login" className="text-sm text-gray-600 hover:text-gray-900 font-medium transition">
            Sign In
          </Link>
          <Link href="/register" className="text-sm bg-indigo-600 text-white font-semibold px-4 py-2 rounded-lg hover:bg-indigo-700 transition">
            Get Started
          </Link>
        </div>
      </nav>

      {/* Hero */}
      <main className="flex-1 flex flex-col items-center justify-center px-4 text-center">
        <div className="max-w-xl animate-fade-in">
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-indigo-50 text-indigo-700 rounded-full text-sm font-medium mb-6">
            <span className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse"></span>
            Powered by RAG + Claude Sonnet
          </div>

          <h1 className="text-5xl font-bold text-gray-900 tracking-tight">
            AI-powered<br />decision engine
          </h1>
          <p className="mt-4 text-xl text-gray-500 leading-relaxed">
            Upload your options. Ask your questions.<br />
            Get one clear, AI-reasoned winner.
          </p>

          <div className="mt-10 flex gap-4 justify-center">
            <Link
              href="/register"
              className="px-6 py-3 bg-indigo-600 text-white font-semibold rounded-lg hover:bg-indigo-700 transition"
            >
              Get Started Free
            </Link>
            <Link
              href="/login"
              className="px-6 py-3 border border-gray-300 text-gray-700 font-semibold rounded-lg hover:bg-gray-100 transition"
            >
              Sign In
            </Link>
          </div>

          <div className="mt-16 grid grid-cols-3 gap-6 text-left">
            {[
              { icon: '📄', title: 'Upload Anything', desc: 'PDF, HTML, images, or plain text. OCR included.' },
              { icon: '🤖', title: 'AI Evaluation', desc: 'RAG pipeline with multi-query, reranking & CRAG.' },
              { icon: '🏆', title: 'Clear Winner', desc: 'Comparative scores + Claude Sonnet verdict.' },
            ].map(f => (
              <div key={f.title} className="bg-white border border-gray-200 rounded-xl p-4">
                <div className="text-2xl mb-2">{f.icon}</div>
                <div className="font-semibold text-gray-800 text-sm">{f.title}</div>
                <div className="text-gray-500 text-xs mt-1">{f.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  )
}

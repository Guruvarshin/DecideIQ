const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

async function request(method, path, body, isFormData = false) {
  const opts = { method, credentials: 'include' }
  if (body) {
    if (isFormData) {
      opts.body = body
    } else {
      opts.headers = { 'Content-Type': 'application/json' }
      opts.body = JSON.stringify(body)
    }
  }
  const res = await fetch(`${BASE}${path}`, opts)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Request failed' }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.status === 204 ? null : res.json()
}

export const api = {
  register:          (email, password)       => request('POST', '/auth/register', { email, password }),
  login:             (email, password)       => request('POST', '/auth/login',    { email, password }),
  logout:            ()                      => request('POST', '/auth/logout'),
  me:                ()                      => request('GET',  '/auth/me'),

  createSession:     (title)                 => request('POST', '/sessions', { title }),
  listSessions:      ()                      => request('GET',  '/sessions'),
  getSession:        (id)                    => request('GET',  `/sessions/${id}`),
  deleteSession:     (id)                    => request('DELETE', `/sessions/${id}`),

  uploadDocument:    (id, formData)          => request('POST', `/sessions/${id}/documents`, formData, true),
  deleteDocument:    (id, idx)               => request('DELETE', `/sessions/${id}/documents/${idx}`),

  addUserQuestions:  (id, questions)         => request('POST', `/sessions/${id}/questions`, { questions }),
  generateQuestions: (id)                    => request('POST', `/sessions/${id}/questions/generate`),
  getQuestions:      (id)                    => request('GET',  `/sessions/${id}/questions`),

  runComparison:     (id)                    => request('POST', `/sessions/${id}/compare`),
  getComparison:     (id)                    => request('GET',  `/sessions/${id}/compare`),

  runEvaluation:     (id, docIdx)            => request('POST', `/sessions/${id}/evaluate?doc_idx=${docIdx}`),
  getEvaluation:     (id)                    => request('GET',  `/sessions/${id}/evaluate`),
}

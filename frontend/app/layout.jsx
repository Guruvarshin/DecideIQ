import './globals.css'

export const metadata = {
  title: 'DecideIQ - AI Decision Engine',
  description: 'Upload your options. Ask your questions. Get one clear winner.',
  icons: {
    icon: '/logo.svg',
    shortcut: '/logo.svg',
  },
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="font-sans antialiased bg-gray-50 text-gray-900 min-h-screen">
        {children}
      </body>
    </html>
  )
}

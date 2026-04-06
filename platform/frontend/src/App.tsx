import { Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { HomePage } from './pages/HomePage'
import { ChartPage } from './pages/ChartPage'
import { OptionsDashboardPage } from './pages/OptionsDashboardPage'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/chart/:symbol" element={<ChartPage />} />
        <Route path="/options" element={<OptionsDashboardPage />} />
      </Routes>
    </Layout>
  )
}

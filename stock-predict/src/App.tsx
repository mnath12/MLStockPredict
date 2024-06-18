import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import StockDashboard from './StockDashboard.tsx'
import { Flex } from '@chakra-ui/react'
import 'vite/modulepreload-polyfill'

function App() {
  const [count, setCount] = useState(0)

  return (
    <StockDashboard />
    
  )
}

export default App

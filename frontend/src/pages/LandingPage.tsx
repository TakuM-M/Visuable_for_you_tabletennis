import { useState, useEffect } from 'react'
import '../lp/dark.css'
import { LpVariation, LpVariationAMobile } from '../lp/lp'

export default function LandingPage() {
  const [isMobile, setIsMobile] = useState(() => window.innerWidth < 768)

  useEffect(() => {
    const fn = () => setIsMobile(window.innerWidth < 768)
    window.addEventListener('resize', fn)
    return () => window.removeEventListener('resize', fn)
  }, [])

  return isMobile ? <LpVariationAMobile /> : <LpVariation />
}

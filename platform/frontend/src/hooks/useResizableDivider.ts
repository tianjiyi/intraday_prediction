import { useState, useRef, useEffect, useCallback } from 'react'

const LS_KEY = 'volumeChartHeight'
const DEFAULT_HEIGHT = 120
const MIN_VOLUME = 60
const MAX_VOLUME = 300
const MIN_MAIN = 200

export function useResizableDivider(
  wrapperRef: React.RefObject<HTMLElement | null>
) {
  const [volumeHeight, setVolumeHeight] = useState(() => {
    const saved = parseInt(localStorage.getItem(LS_KEY) ?? '', 10)
    return isNaN(saved) || saved < MIN_VOLUME ? DEFAULT_HEIGHT : saved
  })

  const isDragging = useRef(false)
  const startY = useRef(0)
  const startHeight = useRef(0)

  useEffect(() => {
    function onMouseMove(e: MouseEvent) {
      if (!isDragging.current || !wrapperRef.current) return
      const delta = startY.current - e.clientY
      const wrapperH = wrapperRef.current.clientHeight
      const maxVol = Math.min(MAX_VOLUME, wrapperH - MIN_MAIN - 8)
      const next = Math.max(MIN_VOLUME, Math.min(maxVol, startHeight.current + delta))
      setVolumeHeight(next)
    }
    function onMouseUp() {
      if (!isDragging.current) return
      isDragging.current = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    document.addEventListener('mousemove', onMouseMove)
    document.addEventListener('mouseup', onMouseUp)
    return () => {
      document.removeEventListener('mousemove', onMouseMove)
      document.removeEventListener('mouseup', onMouseUp)
    }
  }, [wrapperRef])

  // Persist on change
  useEffect(() => {
    localStorage.setItem(LS_KEY, String(volumeHeight))
  }, [volumeHeight])

  const onDividerMouseDown = useCallback((e: React.MouseEvent) => {
    isDragging.current = true
    startY.current = e.clientY
    startHeight.current = volumeHeight
    document.body.style.cursor = 'row-resize'
    document.body.style.userSelect = 'none'
    e.preventDefault()
  }, [volumeHeight])

  return { volumeHeight, onDividerMouseDown }
}

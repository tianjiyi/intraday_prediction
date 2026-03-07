import { useState, useRef, useEffect, useCallback } from 'react'

interface Options {
  storageKey: string
  defaultSize: number
  minSize: number
  maxSize: number
  direction: 'row' | 'col'  // row = vertical drag (height), col = horizontal drag (width)
  /** For row: drag up = grow (volume below chart). For col-left: drag left = grow (right panel). */
  invert?: boolean
}

export function usePanelResize(opts: Options) {
  const { storageKey, defaultSize, minSize, maxSize, direction, invert = false } = opts

  const [size, setSize] = useState(() => {
    const saved = parseInt(localStorage.getItem(storageKey) ?? '', 10)
    return isNaN(saved) || saved < minSize ? defaultSize : Math.min(saved, maxSize)
  })

  const isDragging = useRef(false)
  const startPos = useRef(0)
  const startSize = useRef(0)

  useEffect(() => {
    function onMouseMove(e: MouseEvent) {
      if (!isDragging.current) return
      const delta = direction === 'row'
        ? (invert ? startPos.current - e.clientY : e.clientY - startPos.current)
        : (invert ? startPos.current - e.clientX : e.clientX - startPos.current)
      const next = Math.max(minSize, Math.min(maxSize, startSize.current + delta))
      setSize(next)
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
  }, [direction, invert, minSize, maxSize])

  useEffect(() => {
    localStorage.setItem(storageKey, String(size))
  }, [size, storageKey])

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    isDragging.current = true
    startPos.current = direction === 'row' ? e.clientY : e.clientX
    startSize.current = size
    document.body.style.cursor = direction === 'row' ? 'row-resize' : 'col-resize'
    document.body.style.userSelect = 'none'
    e.preventDefault()
  }, [size, direction])

  return { size, onMouseDown }
}

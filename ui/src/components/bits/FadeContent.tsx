import React, { useEffect, useState } from 'react'
import { cn } from '@/lib/utils'

interface FadeContentProps {
  children: React.ReactNode
  blur?: boolean
  duration?: number
  delay?: number
  className?: string
}

export const FadeContent: React.FC<FadeContentProps> = ({
  children,
  blur = false,
  duration = 400,
  delay = 0,
  className = '',
}) => {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(true)
    }, delay)
    return () => clearTimeout(timer)
  }, [delay])

  return (
    <div
      style={{
        transitionDuration: `${duration}ms`,
      }}
      className={cn(
        'transition-all ease-out',
        isVisible
          ? 'opacity-100 translate-y-0 filter-none'
          : 'opacity-0 translate-y-2',
        blur && !isVisible && 'blur-sm',
        className
      )}
    >
      {children}
    </div>
  )
}

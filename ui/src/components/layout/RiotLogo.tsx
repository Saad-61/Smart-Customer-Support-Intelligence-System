import React from 'react'

interface RiotLogoProps {
  className?: string
}

export const RiotLogo: React.FC<RiotLogoProps> = ({ className = 'w-7 h-7' }) => {
  return (
    <svg
      viewBox="0 0 100 100"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      <rect width="100" height="100" rx="16" fill="#181a1f" />
      {/* Riot Fist Stylized Geometry */}
      <path d="M22 65V43L34 31H45V65H22Z" fill="#d13639" />
      <path d="M47.5 28.5L53.5 22.5H64V65H47.5V28.5Z" fill="#d13639" />
      <path d="M66.5 23.5L72.5 17.5H83V65H66.5V23.5Z" fill="#d13639" />
      <path d="M22 68H83V79H22V68Z" fill="#eb0029" />
      <path d="M18 45.5L28.5 35V65H18V45.5Z" fill="#eb0029" opacity="0.65" />
    </svg>
  )
}

import React from 'react'
import { RiotLogo } from './RiotLogo'

export const Header: React.FC = () => {
  return (
    <header className="border-b border-border bg-[#12141a] px-4 lg:px-8 py-3.5 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <RiotLogo className="w-8 h-8 rounded shrink-0 shadow-sm" />
        <div>
          <div className="text-[11px] font-mono font-semibold tracking-wider text-[#d13639] uppercase">
            Riot Games
          </div>
          <h1 className="text-base lg:text-lg font-bold tracking-tight text-foreground">
            Customer Support Intelligence System
          </h1>
        </div>
      </div>
    </header>
  )
}

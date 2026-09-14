# Riot Games Support Intelligence System — Frontend Dashboard

Web interface for the automated player support ticket triage system.

## Tech Stack
- **Framework**: React 19 + TypeScript + Vite 8
- **UI Components**: Shadcn UI + Radix UI Primitives + Lucide Icons
- **Styling**: Tailwind CSS (Custom Riot Games Dark Slate / Crimson Palette)
- **Charts & Visualizations**: Recharts
- **Animations**: React Bits patterns (`AnimatedNumber`, `FadeContent`)

## Setup & Running Locally

1. **Install Dependencies**:
   ```bash
   npm install
   ```

2. **Environment Configuration**:
   Create a `.env` file (or copy from `.env.example`):
   ```bash
   cp .env.example .env
   ```
   Default backend URL:
   ```env
   VITE_API_URL=http://localhost:8000
   ```

3. **Start Development Server**:
   ```bash
   npm run dev
   ```
   The application will be accessible at `http://localhost:5173`.

4. **Production Build**:
   ```bash
   npm run build
   ```

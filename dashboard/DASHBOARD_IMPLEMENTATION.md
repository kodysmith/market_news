# Dashboard Implementation Guide

This document outlines how to implement the full dashboard frontend.

## Tech Stack

- **Framework**: Next.js 14+ (React)
- **Styling**: Tailwind CSS + Shadcn UI components
- **Data**: Supabase client (read-only)
- **Charts**: Recharts or Chart.js
- **Deployment**: Netlify

## Pages

### 1. Dashboard (`/`)
- Overview cards: Total runs, current portfolio value, today's return
- Recent runs table
- Performance chart (equity curve)
- Current regime indicator

### 2. Runs History (`/runs`)
- Table of all runs with filters
- Sort by date, mode, status
- Click to view run details

### 3. Run Details (`/runs/[id]`)
- Run metadata
- Positions at run completion
- Orders executed
- Regime history for the run

### 4. Positions (`/positions`)
- Current positions across all runs
- Grouped by ticker
- Value and quantity

### 5. Trades (`/trades`)
- All orders across all runs
- Filter by ticker, date, run
- Trade statistics

### 6. Regime (`/regime`)
- Regime detection timeline
- Regime distribution chart
- Confidence trends

## Implementation Steps

1. **Set up Next.js project**
   ```bash
   npx create-next-app@latest dashboard --typescript --tailwind --app
   cd dashboard
   ```

2. **Install dependencies**
   ```bash
   npm install @supabase/supabase-js
   npm install recharts
   npm install @radix-ui/react-*  # For Shadcn components
   ```

3. **Set up Supabase client**
   ```typescript
   // lib/supabase.ts
   import { createClient } from '@supabase/supabase-js'
   
   export const supabase = createClient(
     process.env.NEXT_PUBLIC_SUPABASE_URL!,
     process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
   )
   ```

4. **Create API routes** (if needed)
   - `/api/runs` - Fetch runs
   - `/api/positions` - Fetch positions
   - `/api/orders` - Fetch orders
   - `/api/regime` - Fetch regime history

5. **Build components**
   - `RunCard` - Display run summary
   - `PositionTable` - Display positions
   - `OrderTable` - Display orders
   - `RegimeChart` - Visualize regime history
   - `EquityCurve` - Portfolio value over time

6. **Implement pages**
   - Use Next.js App Router
   - Server components for data fetching
   - Client components for interactivity

## Example Component

```typescript
// app/runs/page.tsx
import { supabase } from '@/lib/supabase'
import RunTable from '@/components/RunTable'

export default async function RunsPage() {
  const { data: runs } = await supabase
    .from('runs')
    .select('*')
    .order('started_at', { ascending: false })
    .limit(100)

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Trading Runs</h1>
      <RunTable runs={runs || []} />
    </div>
  )
}
```

## Styling

Use Shadcn UI components for consistency:
- `Table` for data tables
- `Card` for overview cards
- `Chart` for visualizations
- `Badge` for status indicators

## Deployment

1. Build the project: `npm run build`
2. Deploy to Netlify
3. Set environment variables
4. Enable automatic deployments from Git

## Future Enhancements

- Real-time updates using Supabase subscriptions
- Export data to CSV
- Email alerts for run completions
- Performance analytics
- Comparison with backtest results

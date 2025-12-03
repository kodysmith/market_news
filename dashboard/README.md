# Trading System Dashboard

React/Next.js dashboard for visualizing trading system results from Supabase.

## Setup

```bash
cd dashboard
npm install
```

## Environment Variables

Create `.env.local`:

```
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
```

## Development

```bash
npm run dev
```

## Features

- Dashboard: Overview of recent runs and performance
- History: Historical run results
- Positions: Current positions by run
- Trades: Order history
- Regime: Regime detection history

## Deployment to Netlify

1. Connect your GitHub repository to Netlify
2. Set build command: `npm run build`
3. Set publish directory: `.next`
4. Add environment variables in Netlify dashboard
5. Deploy

## Note

This is a placeholder structure. Full implementation would include:
- React components for each page
- Supabase client integration
- Charts and visualizations
- Real-time updates

See `DASHBOARD_IMPLEMENTATION.md` for detailed implementation guide.

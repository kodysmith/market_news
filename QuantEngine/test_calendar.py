#!/usr/bin/env python3
import sys
from pathlib import Path
import asyncio

# Add QuantEngine root to path
quant_engine_root = Path(__file__).parent
if str(quant_engine_root) not in sys.path:
    sys.path.insert(0, str(quant_engine_root))

async def test_calendar():
    from engine.data_ingestion.live_data_manager import LiveDataManager

    # Create data manager
    config = {}
    manager = LiveDataManager(config)

    # Initialize session
    await manager.initialize()

    # Get calendar events
    calendar_events = manager.get_economic_calendar()
    print(f'Generated {len(calendar_events)} calendar events')

    if calendar_events:
        print('Sample events:')
        for event in calendar_events[:3]:
            print(f'  - {event["title"]} on {event["date"]} ({event["impact"]})')

        # Save to database
        from data_broker import save_calendar_to_db
        save_calendar_to_db(calendar_events)
        print('✅ Saved calendar events to database')
    else:
        print('❌ No calendar events generated')

    await manager.close()

if __name__ == "__main__":
    asyncio.run(test_calendar())


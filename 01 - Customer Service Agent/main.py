import asyncio
from customer_flow import CustomerQueryFlow

async def main():
    flow = CustomerQueryFlow
    await flow.kickoff()


asyncio.run(main())
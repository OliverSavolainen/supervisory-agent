import asyncio
from customer_flow import CustomerQueryFlow

async def run_flow():
    flow = CustomerQueryFlow()
    await flow.kickoff()

async def plot_flow():
    """
    Plot the flow.
    """
    flow = CustomerQueryFlow()
    flow.plot()


def main():
    asyncio.run(run_flow())


def plot():
    asyncio.run(plot_flow())


if __name__ == "__main__":
    main()
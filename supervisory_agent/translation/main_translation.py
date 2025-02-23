import asyncio
from translation_flow import TranslationFlow

async def run_flow():
    flow = TranslationFlow()
    await flow.kickoff()

async def plot_flow():
    """
    Plot the flow.
    """
    flow = TranslationFlow()
    flow.plot()


def main():
    asyncio.run(run_flow())


def plot():
    asyncio.run(plot_flow())


if __name__ == "__main__":
    main()
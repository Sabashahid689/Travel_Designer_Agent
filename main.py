import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from travel_tools import get_fllights, suggest_hotels

# Load environment variables
load_dotenv()

# Initialize the Gemini-compatible client
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Set model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

# Configure run
config = RunConfig(model=model, tracing_disabled=True)

# Agents
destination_agent = Agent(
    name="DestinationAgent",
    instructions="You recommend travel destinations based on the user's mood.",
    model=model
)

booking_agent = Agent(
    name="BookingAgent",
    instructions="You provide flight and hotel information using tools.",
    model=model,
    tools=[get_fllights, suggest_hotels]
)

explore_agent = Agent(
    name="ExploreAgent",
    instructions="You suggest food and places to explore in the destination.",
    model=model
)

def main():
    print("🌍 Welcome to AI Travel Designer!\n")

    mood = input("🧭 What's your travel mood (relaxing, adventure, etc)? → ")

    # Get destination based on mood
    result1 = Runner.run_sync(destination_agent, mood, run_config=config)
    dest = result1.final_output.strip()
    print(f"\n✈️ Destination Suggested: {dest}")

    # Get booking info
    result2 = Runner.run_sync(booking_agent, dest, run_config=config)
    print(f"\n🏨 Booking Info:\n{result2.final_output}")

    # Explore destination
    result3 = Runner.run_sync(explore_agent, dest, run_config=config)
    print(f"\n🍽️ Explore Suggestions:\n{result3.final_output}")

if __name__ == "__main__":
    main()

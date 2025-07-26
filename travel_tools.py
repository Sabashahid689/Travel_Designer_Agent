# travel_tools.py
from agents import function_tool

@function_tool
def get_fllights(destination: str) -> str:
    return f"Flights found to {destination}: PKR 45,000 - PKR 70,000."

@function_tool
def suggest_hotels(destination: str) -> str:
    return f"hotels in {destination}: Pearl Continental, Marriot, Local Guest Houses."
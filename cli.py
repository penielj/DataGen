"""Interactive command-line chat with the DataGen agent."""
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage

load_dotenv()

from datagen.agent import run_agent  # noqa: E402  (load .env before the agent reads it)


def main():
    print("=" * 60)
    print("DataGen Agent - Sample Data Generator")
    print("=" * 60)
    print("Generate sample user data and save to JSON files (in ./output).")
    print()
    print("Examples:")
    print("  - Generate users named John, Jane, Mike and save to users.json")
    print("  - Create users with last names Smith, Jones")
    print("  - Make users aged 25-35 with company.com emails")
    print()
    print("Commands: 'quit' or 'exit' to end")
    print("=" * 60)

    history: List[BaseMessage] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q", ""):
            print("Goodbye!")
            break

        print("Agent: ", end="", flush=True)
        response, _ = run_agent(user_input, history)
        print(response.content)
        print()

        history += [HumanMessage(content=user_input), response]


if __name__ == "__main__":
    main()

from rift.llm.openai_client import OpenAIClient
from rift.memory.history import ConversationMemory
from rift.core.engine import RIFTEngine
from config import RIFTConfig


def main():
    print("RIFT Hallucination Reduction CLI")
    print("================================")

    llm = OpenAIClient()
    memory = ConversationMemory()
    config = RIFTConfig()
    engine = RIFTEngine(llm, memory, config)

    while True:
        user = input("\nYou: ")
        if user.lower() in ("quit", "q", "exit"):
            break

        ans = engine.generate(user)
        print("\nAssistant:", ans)


if __name__ == "__main__":
    main()

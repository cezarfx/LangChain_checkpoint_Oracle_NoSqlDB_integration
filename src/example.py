
import os

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from OracleNoSqlDbTools import OracleNoSqlDbCheckpointer

# Set your own Oracle NoSQL DB endpoint by setting ORACLE_NOSQL_DB_ENDPOINT environment variable. 
ORACLE_NOSQL_DB_ENDPOINT = os.getenv("ORACLE_NOSQL_DB_ENDPOINT", "http://localhost:8080")

# Use your favorite model by setting EXAMPLE_MODEL and EAXMAPL_BASE_URL environment variables,
# this shows a local model served by LM Studio
EXAMPLE_MODEL = os.getenv("EXAMPLE_MODEL", "lfm2.5-1.2b-thinking-mlx")
EXAMPLE_BASE_URL = os.getenv("EXAMPLE_BASE_URL", "http://localhost:1234/v1")

def build_llm():
    # """Create the chatmodel using environment-based configuration."""
    return ChatOpenAI(
        model=EXAMPLE_MODEL,
        base_url=EXAMPLE_BASE_URL,
        #api_key=os.getenv("EXAMPLE_API_KEY", "LLM_PROVIDER_KEY"),
    )


def main() -> None:
    llm = build_llm()
    agent = create_agent(
        model=llm,
        system_prompt=(
            "You are a helpful assistant."
        ),
        checkpointer=OracleNoSqlDbCheckpointer(ORACLE_NOSQL_DB_ENDPOINT, drop_tables=False),
    )

    print("\n=== LLM Invocation 1 ===")
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"What is the answer to the ultimate question?",
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": "thread_1",
                "checkpoint_ns": "t1",
                "checkpoint_id": "checkpoint_1",
                "session_id": "sess_1"
            }
        }
    )

    print("\n  - Response from LLM:")
    print(response["messages"][-1].content)

    # Second invocation to demonstrate retrieval from the database in a different context
    print("\n=== LLM Invocation 2 ===")
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"What is the ultimate question?",
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": "thread_1",
                "checkpoint_ns": "t2",
                "checkpoint_id": "checkpoint_2",
                "session_id": "sess_1"
            }
        }
    )

    print("\n  - LLM 2nd Response ===")
    print(response["messages"][-1].content)

if __name__ == "__main__":
    main()

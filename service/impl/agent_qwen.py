import sys
import os
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

def run_agent():
    # Load config
    config_path = os.path.join(project_root, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Set environment variables for LangChain tools and LLM
    qwen_api_key = config.get("qwen_api_key", "")
    serp_api_key = config.get("serp_api_key", "")
    
    if serp_api_key:
        os.environ["SERPAPI_API_KEY"] = serp_api_key
        
    base_url = config.get("qwen_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    # Initialize the LLM via LangChain's ChatOpenAI (using Qwen's OpenAI compatible API)
    llm = ChatOpenAI(
        model="qwen-plus",
        api_key=qwen_api_key,
        base_url=base_url,
        temperature=0.3
    )
    
    # Load tools (Google Search)
    tools = []
    if serp_api_key:
        try:
            tools = load_tools(["serpapi"], llm=llm)
        except Exception as e:
            print(f"Warning: Failed to load serpapi tool: {e}")
            
    # Initialize the agent with memory using the latest LangChain 0.3/LangGraph V1.0 API
    if tools:
        memory = MemorySaver()
        
        system_prompt = """You are a helpful AI assistant. 
You have access to a Google Search tool (named 'Search'). 
You MUST use this tool to search the internet when the user asks for real-time information, weather, or requests you to 'search google'."""
        
        agent = create_agent(
            llm, 
            tools=tools, 
            checkpointer=memory,
            system_prompt=system_prompt
        )
    else:
        # Fallback if no tools available
        agent = None

    print("Qwen LangChain Agent Ready! Type 'exit' to quit.")

    thread_config = {"configurable": {"thread_id": "main_thread"}}

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        print("\nAssistant: ", end="")
        try:
            if agent:
                response = agent.invoke(
                    {"messages": [HumanMessage(content=user_input)]}, 
                    config=thread_config
                )
                print(response["messages"][-1].content)
            else:
                # Direct LLM call if no tools
                messages = [
                    SystemMessage(content="You are a helpful AI assistant.")
                ]
                messages.append(HumanMessage(content=user_input))
                response = llm.invoke(messages)
                print(response.content)
        except Exception as e:
            print(f"Error during agent execution: {e}")

if __name__ == "__main__":
    run_agent()

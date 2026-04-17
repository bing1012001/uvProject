import os
import json
with open("config.json") as f:
    config = json.load(f)
os.environ["SERPAPI_API_KEY"] = config.get("serp_api_key", "")
qwen_api_key = config.get("qwen_api_key", "")
base_url = config.get("qwen_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")

from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="qwen-plus", api_key=qwen_api_key, base_url=base_url)
tools = load_tools(["serpapi"])
agent = create_react_agent(llm, tools=tools)

response = agent.invoke({"messages": [("user", "What is the weather in New York?")]})
print("Agent Response:", response["messages"][-1].content)
for msg in response["messages"]:
    print(msg)

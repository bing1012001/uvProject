# To run this code you need to install the following dependencies:
# pip install google-genai

from google import genai
from google.genai import types
import json
import os


def call_gemini_completion(prompt: str, temperature: float = 0.7, model: str = "gemini-2.5-pro", role_prompt: str = None, conversation_history: list = None):
    
    api_key = None
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
                api_key = config.get("gemini_api_key")
        except json.JSONDecodeError:
            print("Error: config.json is not a valid JSON file.")
    
    client = genai.Client(
        # api_key="AIzaSyB6O7B_0oA1-dP1a_LDymbV7WbzV2FbVxo",
        api_key=api_key
    )
    
    if conversation_history is None:
        conversation_history = []
        
    contents = []

     # Combine role prompt with user prompt instead of using system role
    if role_prompt:
        current_prompt = f"{role_prompt}\n\nUser: {prompt}"
    else:
        current_prompt = prompt
        
        
    # add conversation history
    for message in conversation_history:
        contents.append(
            types.Content(
                role=message["role"],
                parts=[
                    types.Part.from_text(text=message["content"]),
                ],
            )
        )
    # add current user prompt
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=current_prompt),
            ],
        )
    )
    # print("contents:", contents)
    # print("model:", model)
    # model = "gemini-2.5-pro"
   
    tools = [
        types.Tool(googleSearch=types.GoogleSearch(
        )),
    ]
    generate_content_config = types.GenerateContentConfig(
        # thinking_config = types.ThinkingConfig(
        #     thinking_budget=-1,
        # ),
        temperature=temperature,
        tools=tools,
    )

    try:
        response_stream = client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        full_response_text = ""
        for chunk in response_stream:
            if chunk.text: # Ensure chunk has text content
                print(chunk.text, end="")
                full_response_text += chunk.text
        
        if not full_response_text:
            print("\nNo content received from Gemini. Check API key, model, and network.")
        
        # Add the exchange to conversation history
        conversation_history.extend([
            {"role": "user", "content": current_prompt},
            {"role": "assistant", "content": full_response_text}
        ])
        return conversation_history

    except Exception as e:
        print(f"\nAn error occurred during the API call: {e}")
        # For more specific genai errors:
        # if isinstance(e, genai.APIError):
        #     print(f"API Error details: {e.error_code}, {e.message}")
    
if __name__ == "__main__":
    # Example role prompt for a professional tone
    role_prompt = """You are a highly professional AI assistant. 
    Please maintain a formal and courteous tone in your responses.
    Use technical language when appropriate and always be precise and clear."""
    
    # Casual, friendly tone
    casual_role = """You are a friendly and approachable AI assistant. 
    Use conversational language and feel free to include light humor when appropriate."""

    # Technical expert tone
    technical_role = """You are an expert technical consultant. 
    Provide detailed, technically accurate responses with relevant code examples and best practices."""

    # Educational tone
    educational_role = """You are a patient teacher. 
    reak down complex concepts into simple explanations and use analogies when helpful."""
    
    # result = call_gemini_completion(
    #     prompt="From the Nasdaq market data, tell me the trend for the coming month.",
    #     temperature=0.5,
    #     role_prompt=role_prompt
    # )
    
    # First question
    conversation = call_gemini_completion(  
        prompt="Give me a Chinese peom about AI and technology in the style of Li Bai.",
        temperature=0.5,
        model="gemini-2.0-flash",
        role_prompt=role_prompt
    )
    
    print("\n--- Conversation History ---")
    for msg in conversation:
        print(f"{msg['role'].capitalize()}: {msg['content']}\n")
    print("----------------------------\n")
    
    
    #Follow-up question
    conversation = call_gemini_completion(
        prompt="Can you improve it by adding some elements of nature?",
        temperature=0.5,
        model="gemini-2.0-flash",
        role_prompt=role_prompt,
        conversation_history=conversation
    )
    print("\n--- Conversation History ---")
    for msg in conversation:
        print(f"{msg['role'].capitalize()}: {msg['content']}\n")
    print("----------------------------\n")

# To run this code you need to install the following dependencies:
# pip install openai

from openai import OpenAI
import json
import os


def call_qwen_completion(prompt: str, temperature: float = 0.7, model: str = "qwen-plus", role_prompt: str = None, conversation_history: list = None):
    """
    Call Qwen (Tongyi Qianwen) LLM via DashScope's OpenAI-compatible API.
    
    Args:
        prompt: The user's input prompt
        temperature: Controls randomness (0.0 to 1.0)
        model: The Qwen model to use (e.g., "qwen-turbo", "qwen-plus", "qwen-max")
        role_prompt: System prompt to set the assistant's behavior
        conversation_history: List of previous messages for multi-turn conversation
    
    Returns:
        Updated conversation history list
    """
    
    # Load API configuration from config.json
    api_key = None
    base_url = None
    
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
                api_key = config.get("qwen_api_key")
                base_url = config.get("qwen_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        except json.JSONDecodeError:
            print("Error: config.json is not a valid JSON file.")
    
    if not api_key:
        print("Error: Qwen API key not found in config.json")
        return conversation_history or []
    
    # Initialize OpenAI-compatible client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # Build messages array
    messages = []
    
    # Add system prompt if provided
    if role_prompt:
        messages.append({"role": "system", "content": role_prompt})
    
    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current user message
    messages.append({"role": "user", "content": prompt})
    
    print(f"Using model: {model}")
    print(f"Messages: {messages}")
    
    try:
        # Create streaming completion
        response_stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True
        )
        
        # Collect streamed response
        full_response_text = ""
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                full_response_text += text
        
        print()  # New line after streaming completes
        
        if not full_response_text:
            print("\nNo content received from Qwen. Check API key, model, and network.")
        
        # Update conversation history
        if conversation_history is None:
            conversation_history = []
        
        conversation_history.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": full_response_text}
        ])
        
        return conversation_history

    except Exception as e:
        print(f"\nAn error occurred during the Qwen API call: {e}")
        return conversation_history or []


if __name__ == "__main__":
    # Example role prompts for different tones
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
    Break down complex concepts into simple explanations and use analogies when helpful."""
    
    # First question - Professional tone
    print("=== First Question (Professional Tone) ===")
    conversation = call_qwen_completion(  
        prompt="Give me a Chinese poem about AI and technology in the style of Li Bai.",
        temperature=0.5,
        model="qwen-plus",
        role_prompt=role_prompt
    )
    
    print("\n--- Conversation History ---")
    for msg in conversation:
        print(f"{msg['role'].capitalize()}: {msg['content']}\n")
    print("----------------------------\n")
    
    # Follow-up question - maintains conversation history
    print("=== Follow-up Question ===")
    conversation = call_qwen_completion(
        prompt="Can you improve it by adding some elements of nature?",
        temperature=0.5,
        model="qwen-plus",
        role_prompt=role_prompt,
        conversation_history=conversation
    )
    
    print("\n--- Conversation History ---")
    for msg in conversation:
        print(f"{msg['role'].capitalize()}: {msg['content']}\n")
    print("----------------------------\n")
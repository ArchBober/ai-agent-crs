import os
from dotenv import load_dotenv
from google import genai
import sys


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def main():
    if len(sys.argv) <= 1:
        sys.exit(1)
    elif len(sys.argv[1]) < 1:
        sys.exit(1) 

    print("Hello from ai-agent-crs!")

    client_prompt = sys.argv[1]

    response = client.models.generate_content(
        model='gemini-2.0-flash-001', 
        contents=client_prompt
    )
    
    print(response.text)
    print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
    print(f"Response tokens: {response.usage_metadata.candidates_token_count}")

if __name__ == "__main__":
    main()

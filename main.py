import os
from dotenv import load_dotenv
from google import genai
import sys

from google.genai import types


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def main():
    verbose = False
    arg_count = len(sys.argv)
    if arg_count <= 1:
        sys.exit(1)
    elif len(sys.argv[1]) < 1:
        sys.exit(1)

    user_prompt = sys.argv[1]

    if arg_count > 2:
        if sys.argv[2] == '--verbose':
            verbose = True
        else:
            print(f"Wrong flag {sys.argv[2]}")
            sys.exit(1)

    print("Hello from ai-agent-crs!\n")

    messages = [
        types.Content(role="user", parts=[types.Part(text=user_prompt)]),
    ]

    response = client.models.generate_content(
        model='gemini-2.0-flash-001', 
        contents=messages
    )

    if verbose:
        print(f"User prompt: {messages[0].parts[0].text}")
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}\n")

    print(response.text)



if __name__ == "__main__":
    main()

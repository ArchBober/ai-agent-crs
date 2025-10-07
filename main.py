import os
from dotenv import load_dotenv
from google import genai
from config import SYSTEM_PROMPT
import sys

from google.genai import types
from functions.get_files_info import schema_get_files_info
from functions.get_file_content import schema_get_file_content
from functions.write_file import schema_write_file
from functions.run_python_file import schema_run_python_file


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

    available_functions = types.Tool(
        function_declarations=[
            schema_get_files_info,
            schema_get_file_content,
            schema_write_file,
            schema_run_python_file,
        ],
    )

    config = types.GenerateContentConfig(
        tools=[available_functions],
        system_instruction=SYSTEM_PROMPT
    )


    response = client.models.generate_content(
        model='gemini-2.0-flash-001', 
        contents=messages,
        config=config
    )

    if verbose:
        print(f"User prompt: {messages[0].parts[0].text}")
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}\n")
    if response.function_calls:
        print(f"Calling function: {response.function_calls[0].name}({response.function_calls[0].args})")
    else:
        print(response.text)


if __name__ == "__main__":
    main()

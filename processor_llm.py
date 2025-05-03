from dotenv import load_dotenv
from groq import Groq
import re

load_dotenv()

groq = Groq()

def classify_with_llm(log_message):
    prompt = f"""
        Classify the log message into one of the following categories:
        1. Workflow Error
        2. Deprecation Warning
        If the log message does not fit into any of the above categories, return "Unclassified".
        Only return the category name. No other text i.e. no explanation, no quotes, no preamble.
        Log message: {log_message}"""


    chat_completion=groq.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    response = chat_completion.choices[0].message.content
    formated_response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL).strip()

    return formated_response


if __name__ == "__main__":
    print(classify_with_llm(
        "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."))
    print(classify_with_llm(
        "The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025"))
    print(classify_with_llm("System reboot initiated by user 12345."))
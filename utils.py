from typing import List, Dict, Any
import openai
from openai import OpenAI
import os

openai_api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_api_key

client = OpenAI()


def generate_gpt_first_response(prompt, backstory, current_speaker, current_listener):
    # Construct the messages array based on the input and backstory
    messages: list[dict[str, str | Any] | dict[str, str | Any]] = [
        {"role": "system", "content": "the context of the story is the following:" + f"{backstory}" +
                                      ". Without making any type of comment, without specifying the current speaker,"
                                      " just answer the question (or follow the discussion)  in one or two sentences."
                                      "Remember, you act as you are the current speaker:" + f"{current_speaker.name}"
                                      "and you are talking to the current listener:" + f"{current_listener.name}. Also,"
                                      "the person who has made the first question is:" + f"{current_listener.name}."
                                      "Moreover, take into account that" + f"{current_speaker.name}" "has a level of happiness of"
                                      f"{current_speaker.joy}/100 and a level of anger of {current_speaker.anger}"
         },
        {"role": "user", "content": "the question you have to to answer is:" + f"{prompt}"},
        {"role": "assistant", "content": "Do not repeat the phrase you find in the prompt. When you don't know"
                                         "what to say, just make a random question based on the" + f"{backstory}"
                                         }
    ]

    # API call with the corrected format
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature = 0.7,
        max_tokens = 150
    )
    result = response.choices[0].message.content

    return result


def generate_gpt_following_response(prompt, backstory,conversation_history, current_speaker, current_listener):
    # Construct the messages array based on the input and backstory
    messages: list[dict[str, str | Any] | dict[str, str | Any]] = [
        {"role": "system", "content": "the context of the story is the following:" + f"{backstory}" + "."
                                      "Without making any type of comment, without specifying the current speaker,"
                                      " just answer the question (or follow the discussion) in one or two sentences." 
                                      "The conversation is expected to flow smoothly."
                                      "Remember, you act as you are the current speaker:" + f"{current_speaker.name}"
                                      "and you are talking to the current listener:" + f"{current_listener.name}"
                                      "Moreover, take into account that" + f"{current_speaker.name}" "has a level of"
                                      "anger of" + f"{current_speaker.anger}/100" + "and a level of happyness of" +
                                      f"{current_speaker.joy}/100"},
        {"role": "user", "content": "the question you have to answer is" + f"{prompt}" 
                                    "and is based on this conversation history:" + f"{conversation_history}."},
        {"role": "assistant", "content": "Do not repeat the sentence you find in the prompt. Do not repeat "
                                         "a sentence if it has already used in the conversation. "
                                         "Do not remark any concept. If, you don't know what to answer, just"
                                         "start a random topic which logically fits the information contained here"
                                         "in the context:" + f"{backstory}"
                                         "Whenever possible, answer the questions concisely."}
    ]

    # API call with the corrected format
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature = 0.7,
        max_tokens = 150
    )

    result = response.choices[0].message.content
    return result


def generate_new_response(backstory, conversation_history, current_speaker, current_listener):
    # Construct the messages array based on the input and backstory
    # Construct the messages array based on the input and backstory
    messages: list[dict[str, str | Any] | dict[str, str | Any]] = [
        {"role": "system", "content": "the context of the story is the following:" + f"{backstory}" + "."
                                                                                                      "Without making any type of comment, without specifying the current speaker,"
                                                                                                      " just start a new topic which logically fits the information contained here."
                                                                                                      "Remember, you act as you are the current speaker:" + f"{current_speaker.name}"
                                                                                                                                                            "and you are talking to the current listener:" + f"{current_listener.name}"},
        {"role": "user", "content": "Start a new conversation based on this context:" + f"{backstory}"
                                                                                        "and is based on this conversation history:" + f"{conversation_history}."},
        {"role": "assistant", "content": "Do not repeat any concept contained here" + f"{conversation_history}"}
    ]

    # API call with the corrected format
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )
    result = response.choices[0].message.content

    return result

def generate_new_topic(backstory, conversation_history, current_speaker, current_listener):
    # Construct the messages array based on the input and backstory
    # Construct the messages array based on the input and backstory
    messages: list[dict[str, str | Any] | dict[str, str | Any]] = [
        {"role": "system", "content": "you are the current speaker:" + f"{current_speaker.name}"
                                      "and you are talking to the current listener:" + f"{current_listener.name}."
                                      "Without making any type of comment, without specifying the current speaker,"
                                      "Just act as you do not know what to say and give a new information to the other person"
                                      "Remember, you act as you are the current speaker:" + f"{current_speaker.name}"
                                      "and you are talking to the current listener:" + f"{current_listener.name}"},
        {"role": "user", "content": "Start a new invented topic which is coherent with the context of the backstory:"
                                    + f"{backstory}"},
        {"role": "assistant", "content": "Do not repeat any concept contained here" + f"{conversation_history}"}
    ]

    # API call with the corrected format
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )
    result = response.choices[0].message.content

    return result


def generate_ask(conversation_history, current_speaker, current_listener):
    messages: list[dict[str, str | Any] | dict[str, str | Any]] = [
        {"role": "system", "content": "you are the current speaker:" + f"{current_speaker.name}"
                                      "and you are talking to the current listener:" + f"{current_listener.name}."
                                      "Without making any type of comment, without specifying the current speaker,"
                                      "Just act as you do not know what to say and ask for more information to" + f"{current_listener.name}."
                                      "Remember, you act as you are the current speaker:" + f"{current_speaker.name}"
                                      "and you are talking to the current listener:" + f"{current_listener.name}"},
        {"role": "user", "content": "Ask for more information about the previous question:" + f"{conversation_history[-1]}"},
        {"role": "assistant", "content": "Do not repeat any concept contained here" + f"{conversation_history}"}
    ]

    # API call with the corrected format
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )
    result = response.choices[0].message.content

    return result

def read_initial_message(file_path):
    with open(file_path, 'r') as file:
        message = file.read().strip()
    return message


def memory(prompt, file_path):
    with open(file_path, 'w') as file:
        storage = file.write(prompt)
    return storage


def context(backstory_path):
    with open(backstory_path, 'r') as file:
        message = file.read().strip()
    return message


def resume_conversation(conversation_history):
    # Construct the messages array based on the input and backstory
    messages: list[dict[str, str | Any] | dict[str, str | Any]] = [
        {"role": "system", "content": "Make a resume of the following conversation history"},
        {"role": "user", "content": "Make a resume of the conversation history:" + f"{conversation_history}" +
         "specifying the name of the agents who were speaking and the most important concepts."}
    ]

    # API call with the corrected format
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )
    result = response.choices[0].message.content

    return result


def append_to_file(file_path, new_content):
    """
    Appends new content to an existing file, adding a newline before the new content if the file isn't empty.

    Args:
    - file_path: The path to the file to append to.
    - new_content: The content to append to the file.
    """
    with open(file_path, 'a+') as file:  # Open the file in append mode
        file.seek(0, 2)  # Move the cursor to the end of the file
        if file.tell() > 0:  # Check if the file is not empty
            file.write('\n')  # Add a newline (space line) if the file isn't empty
            file.write('\n')
        file.write(new_content)  # Write the new content

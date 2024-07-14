import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

from utils import *
from BERT import *


class Agent:
    def __init__(self, name):
        self.name = name
        self.conversation_history = []  # Stores the history of the conversation
        self.joy = 0
        self.anger = 80 #random.randint(0, 100)
        self.index_name = "index" + name

    def speak(self, message):
        """Agent sends a message."""
        message_text = f"{self.name} says: {message}"
        print(message_text)
        self.remember(message_text)  # Remember what was said

    def listen(self, message, speaker):
        """Process a message received from another agent."""
        message_text = f"{speaker} says: {message}"
        self.remember(message_text)  # Remember the received message

    def start_conversation(self, message):
        """Start the conversation from the scratch"""
        return self.speak(message)

    def remember(self, message):
        """Remember the conversation"""
        self.conversation_history.append(message)


def dynamic_conversation(agent_a, agent_b, initial_message_path, backstory_path, interactions=5):
    # Initial message is influenced by the backstory but not directly part of the conversational exchange
    backstory = context(backstory_path)
    print(f"Starting Conversation with context: {backstory}")

    initial_message = read_initial_message(initial_message_path)

    conversation_history = [initial_message]

    current_speaker = agent_a  # Let agent_b start the response cycle
    current_listener = agent_b

    current_speaker.start_conversation(initial_message)
    current_listener.listen(initial_message, agent_a.name)

    # Swap roles
    current_speaker, current_listener = current_listener, current_speaker

    for i in range(interactions):
        # The prompt for GPT includes the latest message(s) but not the backstory directly
        prompt = conversation_history[-1]

        # Generate the next message; first iteration can subtly incorporate the backstory as needed
        if i == 0:
            next_message = generate_gpt_first_response(prompt, backstory,current_speaker, current_listener)
        else:
            next_message = generate_gpt_following_response(prompt, backstory,
                                                           current_speaker.conversation_history,
                                                           current_speaker, current_listener)

        current_speaker.speak(next_message)

        current_listener.listen(next_message, current_speaker.name)
        conversation_history.append(next_message)

        # Swap roles
        current_speaker, current_listener = current_listener, current_speaker

    return current_speaker.conversation_history


def dynamic_conversation_vd(agent_a, agent_b, initial_message_path, backstory_path, interactions=10):
    # Initial message is influenced by the backstory but not directly part of the conversational exchange
    backstory = context(backstory_path)
    #print(f"Starting Conversation with context: {backstory}")
    initial_message = read_initial_message(initial_message_path)

    conversation_history = [initial_message]

    current_speaker = agent_a  # Let agent_b start the response cycle
    current_listener = agent_b

    current_speaker.start_conversation(initial_message)
    current_listener.listen(initial_message, agent_a.name)

    # Swap roles
    current_speaker, current_listener = current_listener, current_speaker

    for i in range(interactions):
        # The prompt for GPT includes the latest message(s) but not the backstory directly
        prompt = conversation_history[-1]

        # Generate the next message; first iteration can subtly incorporate the backstory as needed
        if i == 0:
            next_message = generate_gpt_first_response(prompt, backstory,current_speaker, current_listener)
        else:
            new_message = generate_gpt_following_response(prompt, backstory,
                                                           current_speaker.conversation_history,
                                                           current_speaker, current_listener)

            nex_message_emb = get_embeddings(new_message).reshape(1, -1)
            prev_message_emb = get_embeddings(prompt).reshape(1, -1)

            if cosine_similarity(prev_message_emb, nex_message_emb) > 0.9:
                next_message = random.choice([generate_new_response(backstory,conversation_history,current_speaker, current_listener),
                                              generate_new_topic(backstory,conversation_history,current_speaker, current_listener),
                                              generate_ask(conversation_history, current_speaker, current_listener)])
            else:
                next_message = new_message

        current_speaker.speak(next_message)

        current_listener.listen(next_message, current_speaker.name)
        conversation_history.append(next_message)

        # Swap roles
        current_speaker, current_listener = current_listener, current_speaker

    return current_speaker.conversation_history


def dynamic_conversation_memory(agent_a, agent_b, initial_message_path, backstory_path,interactions, init):
    conversation = dynamic_conversation_vd(agent_a, agent_b, initial_message_path, backstory_path, interactions)
    info = resume_conversation(agent_a)
    #append_to_file(file_path='backstory.txt', new_content=info)
    emb = get_embeddings(info).astype(np.float_)

    for ag in [agent_a, agent_b]:
        host = get_index_host(init = init, index_name = ag.index_name)
        index = init.Index(ag.index_name, host = host)
        index.upsert(vectors = [
            {"id": id_conv(index),
             "values": emb}
            ]
        )



import os

from Agent import Agent, dynamic_conversation_memory, dynamic_conversation
from BERT import *
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Initialize agents
agent_a = Agent("aiden")
agent_b = Agent("kai")

p_ak = os.getenv("PINECONE_API_KEY")

# Initialize pinecone
pc = Pinecone(api_key = p_ak)

spec = ServerlessSpec(
        cloud="aws",
        region="us-east-1"
)

if agent_a.index_name not in pc.list_indexes().names():
    pc.create_index(agent_a.index_name, dimension=768, spec = spec)

if agent_b.index_name not in pc.list_indexes().names():
    pc.create_index(agent_b.index_name, dimension=768, spec = spec)


# Path to the file containing the initial message
initial_message_path = "initial_message.txt"
backstory = 'backstory.txt'

# Test
test_path = 'test.txt'


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dynamic_conversation_memory(agent_a, agent_b, initial_message_path, backstory, interactions = 10, init = pc)

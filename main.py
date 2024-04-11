from Agent import Agent, dynamic_conversation_memory, dynamic_conversation


# Initialize agents
agent_a = Agent("Agent A")
agent_b = Agent("Agent B")

# Path to the file containing the initial message
initial_message_path = "initial_message.txt"
backstory = 'backstory.txt'

# Test
test_path = 'test.txt'


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dynamic_conversation_memory(agent_a, agent_b, initial_message_path, backstory)
    dynamic_conversation(agent_a, agent_b, test_path, backstory)

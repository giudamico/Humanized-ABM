# Humanized Agent-based models

This is just the baseline for the project.

In this first demo, two agents start a conversation about a random topic defined in the `initial_message.txt`. At the end of the conversation, one of the two agents is asked to answer a question about the previous conversation to test the memory.

## Actual state

In the actual state of the project, only the cognitive functions have been initialized in the Agent class contained into the `Agent.py`. In particular each agent is endowed with the actions of:

- Speak
- Listen
- Recall a conversation

## Further steps

In the following steps, the agents will be initialized with **emotions**, the final purpose is to make that the future actions of our agents will be affected by their emotions.

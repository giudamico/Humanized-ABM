import numpy as np
from utils import Utils
from sklearn.metrics.pairwise import cosine_similarity
import random

class ConversationManager(Utils):
    def __init__(self, agent_a, agent_b, initial_message_path, backstory_path, interactions=5):
        super().__init__(agent_a, agent_b, initial_message_path, backstory_path, interactions)

    def start_conversation(self):
        backstory = self.read_backstory(self.backstory_path)
        initial_message = self.read_initial_message(self.initial_message_path)
        self.conversation_history.append(initial_message)

        self.current_speaker.start_conversation(initial_message)
        self.current_listener.listen(initial_message, self.current_speaker.name)

        # Swap roles
        self.current_speaker, self.current_listener = self.current_listener, self.current_speaker

    def continue_conversation(self):
        backstory = self.read_backstory(self.backstory_path)
        for i in range(self.interactions):
            prompt = self.conversation_history[-1]

            if i == 0:
                if self.is_question(prompt):
                    memory_answer = self.retrieve_from_pinecone_memory(prompt)
                    if memory_answer:
                        print("YEAH")
                        next_message = memory_answer
                    else:
                        next_message = self.generate_gpt_first_response(prompt, backstory,self.current_listener, self.current_speaker)
            else:
                if self.is_question(prompt):
                    memory_answer = self.retrieve_from_pinecone_memory(prompt)
                    if memory_answer:
                        print("YEAH")
                        new_message = memory_answer

                else:
                    new_message = self.generate_gpt_following_response(prompt, backstory,self.conversation_history,self.current_listener, self.current_speaker)

                new_message_emb = self.get_embedding(new_message)
                prev_message_emb = self.get_embedding(prompt)

                if cosine_similarity(prev_message_emb, new_message_emb) > 0.9:
                    next_message = random.choice([
                        self.follow_conversation(backstory,self.conversation_history,prompt,self.current_speaker,self.current_listener),
                        self.generate_new_topic(backstory,self.conversation_history,self.current_speaker,self.current_listener),
                        self.generate_ask(self.conversation_history,self.current_speaker,self.current_listener)
                    ])
                else:
                    next_message = new_message

            # Evaluate the sentiment of the new message
            sentiment_label, sentiment_score = self.evaluate_sentiment(next_message)
            objective_value = self.objective_function(sentiment_label, sentiment_score)

            self.current_speaker.speak(next_message)
            self.current_listener.listen(next_message, self.current_speaker.name)
            self.conversation_history.append(next_message)

            if self.is_question(prompt) and not self.retrieve_from_pinecone_memory(prompt):
                emb = self.get_embedding(prompt).astype(np.float_)
                emb_list = emb.flatten().tolist()

                host = self.get_index_host(self.pinecone, self.current_speaker.index_name)
                index = self.pinecone.Index(self.current_speaker.index_name, host=host)
                index.upsert(vectors=[{
                    "id": self.id_conv(index),
                    "values": emb_list,
                    "metadata": {"text": next_message}
                }])

                # Swap roles
            self.current_speaker, self.current_listener = self.current_listener, self.current_speaker

    def dynamic_conversation_memory(self):
        self.start_conversation()
        self.continue_conversation()
        #info = self.resume_conversation(self.conversation_history)
        #emb = self.get_embedding(info).astype(np.float_)

        #emb_list = emb.flatten().tolist()       #Flatten ensures that the embedding is a flat list (not a nested list) and explicitly converts it to a list of floats.
        #if not all(isinstance(x, float) for x in emb_list):
        #    raise ValueError("Embedding contains non-float values")

        #for agent in [self.agent_a, self.agent_b]:
        #    host = self.get_index_host(self.pinecone,agent.index_name)
        #    index = self.pinecone.Index(agent.index_name, host=host)
        #    index.upsert(vectors=[{
        #        "id": self.id_conv(index),
        #        "values": emb_list,
        #        "metadata": {"text": info,
        #                      "anger": anger_level,
        #                       "joy": joy_level}
        #    }])


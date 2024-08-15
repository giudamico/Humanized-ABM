import os
from pinecone import Pinecone
import torch
from transformers import BertTokenizer, BertModel, pipeline
from sklearn.preprocessing import normalize
import numpy as np
class ConversationManagerBase:
    def __init__(self, agent_a, agent_b, initial_message_path, backstory_path, interactions=5):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.initial_message_path = initial_message_path
        self.backstory_path = backstory_path
        self.interactions = interactions
        self.conversation_history = []
        self.current_speaker = agent_a
        self.current_listener = agent_b

        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key is not set in the environment variables.")
        pinecone = Pinecone(api_key=self.pinecone_api_key, environment='us-west1-gcp')
        self.pinecone = pinecone

        # Load BERT model and tokenizer once during initialization
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

        # Load sentiment analysis pipeline
        self.sentiment_pipeline = pipeline("sentiment-analysis")

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = embeddings.numpy().reshape(1,-1)

        return embeddings

    def evaluate_sentiment(self, text):
        result = self.sentiment_pipeline(text)
        return result[0]['label'], result[0]['score']

    def objective_function(self, sentiment_label, sentiment_score):
        # Assuming the goal is to maximize happiness (positive sentiment)
        if sentiment_label == 'POSITIVE':
            return sentiment_score
        elif sentiment_label == 'NEGATIVE':
            return -sentiment_score
        else:
            return 0  # Neutral sentiment or unrecognized label

    @staticmethod
    def is_question(text):
        # Simple heuristic to check if the text is a question
        return text.strip().endswith('?')

    def retrieve_from_pinecone_memory(self, prompt):
        # Get the embedding for the prompt
        prompt_emb = self.get_embedding(prompt).astype(np.float_)
        prompt_emb = prompt_emb.flatten().tolist()
        # Normalize the embedding
        #prompt_emb = normalize([prompt_emb])[0].tolist()

        # Query Pinecone to find the most similar embeddings

        index_name = self.agent_a.index_name  # Assuming both agents share the same memory index
        host = self.get_index_host(self.pinecone, index_name)
        index = self.pinecone.Index(index_name, host=host)

        query_result = index.query(vector=prompt_emb, top_k=1, include_metadata=True)

        if query_result['matches']:
            return query_result['matches'][0]['metadata']['text']
        else:
            return None

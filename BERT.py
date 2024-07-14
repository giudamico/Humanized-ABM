import transformers
from transformers import BertTokenizer, BertModel
import pinecone
import torch
import os

#Pinecone stage
pinecone_api_key = os.getenv("PINECONE_API_KEY")
dimension = 768

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    # Use the mean of the token embeddings as the sentence embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

def id_conv(index):
    index_stats = index.describe_index_stats()
    return f"Conv. {index_stats['total_vector_count']}"


def get_index_host(init, index_name):
    # Describe the index to get its details
    index_description = init.describe_index(index_name)

    # Extract the host information
    host = index_description.host
    return host


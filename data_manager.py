from dotenv import load_dotenv
import os
from pymongo import MongoClient
import json
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding



class DataManager:
    def __init__(self, mongo_uri, db_name, collection_name, json_path=None):
        print("Initializing DataManager...")
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.json_path = json_path
        self.parser = SentenceSplitter()

    def load_json_data(self):
        print(f"Loading data from {self.json_path}...")
        with open(self.json_path, 'r') as file:
            return json.load(file)

    def create_documents(self, faq_data):
        print("Creating documents...")
        documents = []
        for item in faq_data['questions']:
            document = Document(
                text=f"Question: {item['question']}\nAnswer: {item['answer']}",
                metadata={'question': item['question'], 'answer': item['answer']},
                text_template="Metadata: {metadata_str}\n-----\nContent: {content}"
            )
            documents.append(document)
        return documents

    def process_documents(self):
        faq_data = self.load_json_data()
        documents = self.create_documents(faq_data)
        for document in documents:
            self.collection.insert_one({
                'text': document.text,
                'metadata': document.metadata
            })

class DocumentProcessor:
    def __init__(self, data_manager, embedding_service):
        print("Initializing DocumentProcessor...")
        self.data_manager = data_manager
        self.embedding_service = embedding_service

    def process_documents(self):
        print("Processing documents...")
        documents = self.data_manager.create_documents(self.data_manager.load_json_data())
        nodes = []
        for document in documents:
            print(f"Getting embedding for document content: {document.text[:60]}...")
            embedding = self.embedding_service.get_text_embedding(document.text)
            document.embedding = embedding  # Set the embedding attribute
            document.metadata['embedding'] = ",".join(map(str, embedding))  # Convert embedding to string
            nodes.append(document)
        return nodes

class IndexManager:
    def __init__(self, mongo_uri, db_name, collection_name, index_name):
        print("Initializing IndexManager...")
        self.client = MongoClient(mongo_uri)
        self.vector_store = MongoDBAtlasVectorSearch(self.client, db_name=db_name, collection_name=collection_name, index_name=index_name)

    def add_to_index(self, nodes):
        print("Adding nodes to index...")
        self.vector_store.add(nodes)

    def create_index(self):
        print("Creating vector store index...")
        return VectorStoreIndex.from_vector_store(self.vector_store)


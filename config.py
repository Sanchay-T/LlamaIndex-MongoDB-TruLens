from dotenv import load_dotenv
import os
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding



class Configuration:
    def __init__(self):
        print("Loading environment variables...")
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            print("ERROR: OpenAI API key not found.")
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=256)
        self.llm = OpenAI(api_key=self.openai_api_key)
        self.configure()

    def configure(self):
        print("Configuring settings...")
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
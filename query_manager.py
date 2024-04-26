from llama_index.core.response.notebook_utils import display_response
import pprint

class QueryManager:
    def __init__(self, index):
        self.query_engine = index.as_query_engine(similarity_top_k=3)

    def perform_query(self, query):
        response = self.query_engine.query(query)
        display_response(response)
        pprint.pprint(response.source_nodes)
        return response

from data_manager import DataManager, DocumentProcessor, IndexManager
from config import Configuration
def setup():
    config = Configuration()
    mongo_uri = 'mongodb+srv://sanchayt:xerxes101@cluster0.rab7pym.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
    db_name = 'trial'
    collection_name = 'qq'
    json_path = 'Ecommerce_FAQ_Chatbot_dataset.json'

    data_manager = DataManager(mongo_uri, db_name, collection_name, json_path=json_path)
    document_processor = DocumentProcessor(data_manager, config.embed_model)
    nodes = document_processor.process_documents()

    index_manager = IndexManager(mongo_uri, db_name, collection_name, index_name='vector_index')
    index_manager.add_to_index(nodes)
    vector_index = index_manager.create_index()

    return vector_index

from flask import Flask
import logging
import secrets
import os
from .llm import initialize_llm
from langchain.embeddings import HuggingFaceBgeEmbeddings


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = secrets.token_hex(16)
    app.config["MONGO_URI"] = "mongodb://mongo:27017/documents_db"
    os.environ["OPENAI_API_KEY"] = "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"
    app.topics = ['ART', 'EXP']
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    app.logger = logging.getLogger(__name__)
    app.docsearch = None
    model_name = "BAAI/bge-base-en"
    encode_kwargs = {'normalize_embeddings': True,
                     'device': 'cpu'}
    app.embeddings = HuggingFaceBgeEmbeddings(model_name=model_name,
                                          encode_kwargs=encode_kwargs)


    from flask_pymongo import PyMongo
    app.mongo = PyMongo(app)

    # app.the_llm = initialize_llm('google/flan-t5-large',  ll_model_type='local')
    app.the_llm = initialize_llm('gpt-3.5-turbo')

    from .documents import documents_bp
    app.register_blueprint(documents_bp)

    from .images import images_bp
    app.register_blueprint(images_bp)

    from .conversations import conversations_bp
    app.register_blueprint(conversations_bp)

    return app

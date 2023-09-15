import os
import pickle
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy
from langchain.embeddings.openai import OpenAIEmbeddings
from werkzeug.utils import secure_filename

from flask import current_app as app
from .image_encoder import ImageEncoder
# from .test_img import register_embeddings



def load_pdfs(documents_dir):
    app.logger.info("------------------------->>>>>>>>")
    app.logger.info("load_pdfs")
    app.logger.info("------------------------->>>>>>>>")
    # loader = DirectoryLoader(documents_dir)
    loader = DirectoryLoader(path=documents_dir,
                             show_progress=True,
                             use_multithreading=True,
                             loader_cls=TextLoader)
    documents = loader.load()
    return documents



def load_docs(documents_dir, _file):
    with open(os.path.join(documents_dir, _file), 'rb') as f:
        documents = pickle.load(f)
    return documents

def load_docs2(uploads_dir, _file):
    file = _file[0]
    files = []
    if file.filename.lower().endswith(('.pkl', '.pickle')):
        filename = secure_filename(file.filename)
        file_path = os.path.join(uploads_dir, filename)
        docs = load_docs(uploads_dir, filename)
        app.logger.info("------------------------->>>>>>>>")
        app.logger.info("inside gen_txt_files_for_docs")
        app.logger.info(f" file -----> {filename}")
        app.logger.info(f" docs -----> {docs}")
        app.logger.info("------------------------->>>>>>>>")
        for doc in docs:
            page = doc.page_content
            meta = doc.metadata
            cid = meta['contentId']
            app.logger.info("------------------------->>>>>>>>")
            app.logger.info("inside file store for pickle")
            app.logger.info(f"registering file -----> {meta['contentId']}")
            app.logger.info("------------------------->>>>>>>>")
            with open(os.path.join(uploads_dir, f"{cid}.txt"), 'w') as f:
                f.write(page)
                files.append(f"{cid}.txt")
        return files
    else:
        return None

def preprocess_documents(uploads_dir, topic, _type='pdfs', _file=None):
    postgres_user = os.getenv('DB_USER')
    postgres_password = os.getenv('DB_PASSWORD')
    postgres_db = os.getenv('DB_NAME')
    pgvector_host = os.getenv('DB_HOST')
    pgvector_port = os.getenv('DB_PORT')

    connection_string = f'postgresql://{postgres_user}:{postgres_password}@{pgvector_host}:{pgvector_port}/{postgres_db}'

    documents = None
    if _type == 'pdfs':
        documents = load_pdfs(uploads_dir)
    elif _type == 'docs':
        app.logger.info("------------------------->>>>>>>>")
        app.logger.info("processing files")
        app.logger.info(f"Files to upload {uploads_dir}")
        app.logger.info(f"Files to upload {_file}")
        app.logger.info("------------------------->>>>>>>>")
        file_name = _file[0].filename
        documents = load_docs(uploads_dir, file_name)
        app.logger.info("------------------------->>>>>>>>")
        app.logger.info("processing files 2")
        app.logger.info(f"Files to upload {_file}")
        app.logger.info(f"Documents {documents}")
        app.logger.info("------------------------->>>>>>>>")

    if documents is not None:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

    # Loading the embeddings
        embeddings = app.embeddings #OpenAIEmbeddings()
        app.logger.info("------------------------->>>>>>>>")
        app.logger.info("processing files 3")
        app.logger.info("------------------------->>>>>>>>")
        try:

            vectordb = PGVector.from_documents(documents=texts,
                                               embedding=embeddings,
                                               collection_name=topic,
                                               connection_string=connection_string)
            return True
        except Exception as e:
            app.logger(f"Error: registering embeddings")
            return False
    else:
        app.logger("Error: No documents to process")
        return False

def preprocess_images(uploads_dir, topic, logger):
    postgres_user = os.getenv('DB_USER')
    postgres_password = os.getenv('DB_PASSWORD')
    postgres_db = os.getenv('DB_NAME')
    pgvector_host = os.getenv('DB_HOST')
    pgvector_port = os.getenv('DB_PORT')

    connection_string = f'postgresql://{postgres_user}:{postgres_password}@{pgvector_host}:{pgvector_port}/{postgres_db}'

    encoder = ImageEncoder(uploads_dir)
    encoder.load_images()
    img_embed = encoder.image_encoder_pair()

    try:
        # register_embeddings(connection_str=connection_string, embeddings=img_embed, logger=logger)
        # vectordb = IPGVector.from_embeddings(img_embed,
        #                                    encoder,
        #                                    collection_name=topic,
        #                                    connection_string=connection_string,
        #                                     logger=app.logger)
        return True
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

    return None
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask import send_from_directory
from datetime import datetime
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from flask import current_app as app
import os
from .preprocess import preprocess_documents, load_docs
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import magic


loaded_env = load_dotenv(find_dotenv())

documents_bp = Blueprint('documents', __name__, url_prefix='/documents')

FTYPES = ('.pdf', '.doc', '.docx', '.ppt', '.pptx', '.txt', '.pkl', '.pickle')


def get_files_in_directory(directory_path):
    files = []
    mime = magic.Magic(mime=True)
    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            file_path = os.path.join(directory_path, filename)
            files.append((file_path, filename, mime.from_buffer(file_path)))
    return files


def fetch_files(directory_path):
    file_paths = get_files_in_directory(directory_path)
    file_storage_objects = []

    for file_path in file_paths:
        # with open(file_path[0], 'rb') as f:
        file_storage = FileStorage(stream=open(file_path[0], 'rb'), filename=str(file_path[1]), content_type=file_path[2])
        file_storage_objects.append(file_storage)
    return file_storage_objects

def files_store_in_folder(files, uploads_dir):
    for file in files:
        if file and file.filename.lower().endswith(FTYPES):
            filename = secure_filename(file.filename)
            file_path = os.path.join(uploads_dir, filename)
            file.save(file_path)

def move_files_and_register_uploads(files, topic, topic_dir, uploads_dir):
    for file in files:
        filename = secure_filename(file.filename)
        old_file_path = os.path.join(uploads_dir, filename)
        new_file_path = os.path.join(topic_dir, filename)
        os.rename(old_file_path, new_file_path)

        # Save document information in the database
        document = {
            'topic': topic,
            'document_name': filename,
            'upload_datetime': datetime.utcnow(),
            'uri': new_file_path
        }
        app.mongo.db.documents.insert_one(document)

def gen_txt_files_for_docs(_file, uploads_dir):
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

def move_files_and_register_docs(files, topic, topic_dir, uploads_dir):
    app.logger.info("------------------------->>>>>>>>")
    app.logger.info("move_files_and_register_docs")
    app.logger.info(f"registering file -----> {files}")
    app.logger.info(f"topic -----> {topic}")
    app.logger.info(f"topic dir -----> {topic_dir}")
    app.logger.info(f"upload dir -----> {uploads_dir}")
    app.logger.info("------------------------->>>>>>>>")
    files = gen_txt_files_for_docs(files, uploads_dir)
    if files is not None:
        for file in files:
            old_file_path = os.path.join(uploads_dir, file)
            new_file_path = os.path.join(topic_dir, file)
            os.rename(old_file_path, new_file_path)

            # Save document information in the database
            document = {
                'topic': topic,
                'document_name': file,
                'upload_datetime': datetime.utcnow(),
                'uri': new_file_path
            }
            app.mongo.db.documents.insert_one(document)
    else:
        app.logger.info("------------------------->>>>>>>>")
        app.logger.info("ERROR SAVING TXT FROM DOCS AND REGISTERING IN MONGO")
        app.logger.info("------------------------->>>>>>>>")

@documents_bp.route('/')
def documents():
    documents = list(app.mongo.db.documents.find().sort([('upload_datetime', -1)]))
    return render_template('documents.html', documents=documents)

@documents_bp.route('/doc_proc', methods=['GET', 'POST'])
def start_populating_doc_db():
    topics = app.topics
    uploads_dir = os.getenv('UPLOADS_DIR')
    os.makedirs(uploads_dir, exist_ok=True)

    if request.method == 'POST':
        topic = request.form.get('topic')
        files = request.files.getlist('files')
        topic_dir = os.path.join(uploads_dir, topic)

        app.logger.info("------------------------->>>>>>>>")
        app.logger.info("inside start")
        app.logger.info(os.getenv('BASE_DOCUMENTS'))
        app.logger.info(f"topic selected {topic}")
        app.logger.info(f"Files to upload {len(files)}")
        app.logger.info(f"Files to upload {files}")
        app.logger.info("------------------------->>>>>>>>")
        files_store_in_folder(files, uploads_dir)

        if preprocess_documents(uploads_dir, topic, _type='docs', _file=files):
            move_files_and_register_docs(files, topic, topic_dir, uploads_dir)
        else:
            print("Error processing documents")
            # @todo add error to UI and do something

        return redirect(url_for('documents.documents'))

    return render_template('start_base.html', topics=topics)



@documents_bp.route('/add', methods=['GET', 'POST'])
def add_documents():
    topics = app.topics
    uploads_dir = os.getenv('UPLOADS_DIR')
    os.makedirs(uploads_dir, exist_ok=True)

    if request.method == 'POST':
        topic = request.form.get('topic')
        files = request.files.getlist('files')
        topic_dir = os.path.join(uploads_dir, topic)
        os.makedirs(topic_dir, exist_ok=True)
        app.logger.info("------------------------->>>>>>>>")
        app.logger.info("add")
        app.logger.info(f"Files to upload {files}")
        app.logger.info("------------------------->>>>>>>>")
        files_store_in_folder(files, uploads_dir)

        if preprocess_documents(uploads_dir, topic):
            move_files_and_register_uploads(files, topic, topic_dir, uploads_dir)
        else:
            app.logger.info("------------------------->>>>>>>>")
            app.logger.info("ERROR PROCESSING DOCUMENTS")
            app.logger.info("------------------------->>>>>>>>")
            # @todo add error to UI and do something

        return redirect(url_for('documents.documents'))

    return render_template('add_documents.html', topics=topics)





@documents_bp.route('/uploads/<topic>/<path:filename>')
def serve_uploaded_file(topic, filename):
    uploads_dir = os.path.join(os.getenv('UPLOADS_DIR'), topic)
    return send_from_directory(uploads_dir, filename, as_attachment=True)

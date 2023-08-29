from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask import send_from_directory
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import current_app as app
import os
from .preprocess import preprocess_documents, preprocess_images, load_pdfs
from dotenv import find_dotenv, load_dotenv

loaded_env = load_dotenv(find_dotenv())

images_bp = Blueprint('images', __name__, url_prefix='/images')


@images_bp.route('/')
def images():
    images = list(app.mongo.db.images.find().sort([('upload_datetime', -1)]))
    return render_template('images.html', images=images)


@images_bp.route('/add', methods=['GET', 'POST'])
def add_images():
    topics = app.topics
    uploads_dir = os.getenv('UPLOAD_IMG_DIR')
    os.makedirs(uploads_dir, exist_ok=True)

    if request.method == 'POST':
        topic = request.form.get('topic')
        files = request.files.getlist('files')
        topic_dir = os.path.join(uploads_dir, topic)
        os.makedirs(topic_dir, exist_ok=True)
        for file in files:
            if file and file.filename.lower().endswith(('.png', '.jpg')):
                filename = secure_filename(file.filename)
                file_path = os.path.join(uploads_dir, filename)
                file.save(file_path)
                old_file_path = os.path.join(uploads_dir, filename)
                new_file_path = os.path.join(topic_dir, filename)
                os.rename(old_file_path, new_file_path)

        preprocess_images(uploads_dir, topic, app.logger)

    #     if preprocess_pdf(uploads_dir, topic):
    #         for file in files:
    #             filename = secure_filename(file.filename)
    #             old_file_path = os.path.join(uploads_dir, filename)
    #             new_file_path = os.path.join(topic_dir, filename)
    #             os.rename(old_file_path, new_file_path)
    #
    #             # Save document information in the database
    #             document = {
    #                 'topic': topic,
    #                 'image_name': filename,
    #                 'upload_datetime': datetime.utcnow(),
    #                 'uri': new_file_path
    #              }
    #             app.mongo.db.documents.insert_one(document)
    #     else:
    #         print("Error processing documents")
    #         # @todo add error to UI and do something
    #
    #     return redirect(url_for('images.images'))
    #
    return render_template('add_images.html', topics=topics)


@images_bp.route('/uploads/<topic>/<path:filename>')
def serve_uploaded_file(topic, filename):
    uploads_dir = os.path.join(os.getenv('UPLOADS_DIR'), topic)
    return send_from_directory(uploads_dir, filename, as_attachment=True)

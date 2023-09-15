from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask import send_from_directory
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import current_app as app
import os
from .preprocess import preprocess_documents, preprocess_images, load_pdfs
from dotenv import find_dotenv, load_dotenv
from flask import current_app as app
import csv
import json
from .myutils import files_store_in_folder
import shutil

import torch
from PIL import Image
from torchvision import transforms, models
import torchvision
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import sqlalchemy
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy_utils import database_exists, create_database
import uuid
from sqlalchemy.dialects.postgresql import JSON, UUID
from pgvector.sqlalchemy import Vector


loaded_env = load_dotenv(find_dotenv())

images_bp = Blueprint('images', __name__, url_prefix='/images')

FTYPES = ('.csv', '.tsv', '.txt', '.pkl', '.pickle')


#########
# Define a custom image embedding function
#########

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet18(weights='DEFAULT')
model.fc = torch.nn.Identity()
model.eval()


def image_embedding(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    # Generate embeddings
    with torch.no_grad():
        output = model(input_batch)

    return output.detach().numpy()[0]

##############################################################################
Base = declarative_base()

class BaseModel(Base):
    __abstract__ = True
    uuid = sqlalchemy.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

class FeatureStore(BaseModel):
    __tablename__ = "images_features"

    features: Vector = sqlalchemy.Column(Vector(None))
    image_path =  sqlalchemy.Column(sqlalchemy.String, nullable=True)
    image_genre =  sqlalchemy.Column(sqlalchemy.String, nullable=True)
    image_style = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    topic = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    artistContentId =  sqlalchemy.Column(sqlalchemy.String, nullable=True)
    cmetadata = sqlalchemy.Column(JSON, nullable=True)

class QueryResult:
    FeatureStore: FeatureStore
    distance: float

###################################################################################

def start_database_pgv():
    postgres_user = os.getenv('DB_USER')
    postgres_password = os.getenv('DB_PASSWORD')
    postgres_db = os.getenv('DB_NAME')
    pgvector_host = os.getenv('DB_HOST')
    pgvector_port = os.getenv('DB_PORT')
    url = URL.create(
        drivername="postgresql",
        username=postgres_user,
        password=postgres_password,
        host=pgvector_host,
        port=pgvector_port,
        database=postgres_db
    )
    engine = create_engine(url)
    if not database_exists(engine.url):
        create_database(engine.url)

    Base.metadata.create_all(engine)

    connection = engine.connect()

    Session = sessionmaker(bind=engine)
    session = Session()

    statement = sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector")
    session.execute(statement)
    session.commit()
    return session

def load_imgs_file(file_paths):
    img_data = []
    for file_path in file_paths:
        with open(file_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_data.append(json.loads(json.dumps(row)))
    return img_data

def data_json(arr):
    jd = []
    for ele in arr:
        jd.append(json.dumps(ele))
    return jd

@images_bp.route('/list', methods=['GET', 'POST'])
def list_arts():
    wkpd_files = "/app/base_data/imgs/dimgs_0.csv"
    dimgs = load_imgs_file(wkpd_files)
    meta = json.loads(dimgs[0]['app_meta'])


    app.logger.info("------------------------->>>>>>>>")
    app.logger.info("inside gen_txt_files_for_docs")
    app.logger.info(f" ids -----> {dimgs}")
    app.logger.info(f" ids -----> {meta['genre']}")
    app.logger.info("------------------------->>>>>>>>")





@images_bp.route('/')
def images():
    images = list(app.mongo.db.images.find().sort([('upload_datetime', -1)]))
    app.logger.info("------------------------->>>>>>>>")
    app.logger.info("inside imagenes /")
    app.logger.info(f" imagenes array -----> {images}")
    app.logger.info("------------------------->>>>>>>>")
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
        files_store_in_folder(files, uploads_dir, FTYPES)

        file_paths = [os.path.join(uploads_dir, f.filename) for f in files]

        app.logger.info("------------------------->>>>>>>>")
        app.logger.info("inside imagenes /add")
        app.logger.info(f" file -----> {file_paths}")
        app.logger.info("------------------------->>>>>>>>")

        meta_loaded = load_imgs_file(file_paths)
        images = meta_loaded
        app.logger.info("------------------------->>>>>>>>")
        app.logger.info("inside imagenes /")
        app.logger.info(f" imagenes array -----> {len(images)}")
        app.logger.info("------------------------->>>>>>>>")


        session = start_database_pgv()

        COPY_FILE = False
        imgs_store = []
        for file in images:
            if file['include'] == 'False':
                old_file_path = os.path.join(file['app_path'], f"{file['contentId']}.jpg")
                if COPY_FILE:
                    new_file_path = os.path.join(topic_dir, f"{file['contentId']}.jpg")
                    shutil.copyfile(src=old_file_path, dst=new_file_path)
                meta = json.loads(file['app_meta'])
                imgs_store.append((old_file_path, meta))

        app.logger.info("------------------------->>>>>>>>")
        app.logger.info("inside imagenes /")
        app.logger.info(f" imagenes files paths -----> {len(imgs_store)}")
        app.logger.info("------------------------->>>>>>>>")

        for data in imgs_store:
            embeddings = image_embedding(data[0])
            metad = data[1]
            app.logger.info("------------------------->>>>>>>>")
            app.logger.info(f" data path -----> {data[0]}")
            app.logger.info(f" data metadata -----> {metad}")
            app.logger.info("------------------------->>>>>>>>")

            try:
                imgf = FeatureStore(features = embeddings,
                                    image_path = data[0],
                                    image_genre = metad['genre'],
                                    image_style = metad['style'],
                                    topic = topic,
                                    artistContentId = metad['artistContentId'],
                                    cmetadata = metad)
                session.add(imgf)
                session.commit()
            except Exception as e:
                app.logger.info(f"Error: registering embeddings in pgvector")

            # Save document information in the database
            try:
                document = {
                    'topic': topic,
                    'image_path': data[0],
                    'upload_datetime': datetime.utcnow(),
                    'uri': metad
                }
                app.mongo.db.images.insert_one(document)
            except Exception as e:
                app.logger.info(f"Error: registering embeddings in mongo")

        return render_template('images.html', images=images)
    #
    return render_template('add_images.html', topics=topics)


@images_bp.route('/uploads/<topic>/<path:filename>')
def serve_uploaded_file(topic, filename):
    uploads_dir = os.path.join(os.getenv('UPLOADS_DIR'), topic)
    return send_from_directory(uploads_dir, filename, as_attachment=True)

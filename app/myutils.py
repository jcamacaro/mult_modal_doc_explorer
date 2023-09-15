from werkzeug.utils import secure_filename
import os

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import sqlalchemy
from sqlalchemy.dialects.postgresql import JSON, UUID
from pgvector.sqlalchemy import Vector
from sqlalchemy_utils import database_exists, create_database



def files_store_in_folder(files, uploads_dir, _FTYPES):
    for file in files:
        if file and file.filename.lower().endswith(_FTYPES):
            filename = secure_filename(file.filename)
            file_path = os.path.join(uploads_dir, filename)
            file.save(file_path)

def copy_immgs_process(files):
    for file in files:
        filename = f"{file['app_path']}/{file['contentId']}.jpg"
        filename = secure_filename(filename)
        old_file_path = os.path.join(uploads_dir, filename)
        new_file_path = os.path.join(topic_dir, filename)
        os.rename(old_file_path, new_file_path)

################################################################

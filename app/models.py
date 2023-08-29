from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    text_file_path = db.Column(db.String(255))

    def __repr__(self):
        return f"<Document {self.id}: {self.name}>"

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    image_file_path = db.Column(db.String(255))

    def __repr__(self):
        return f"<Image {self.id}: {self.name}>"

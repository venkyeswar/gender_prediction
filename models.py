from db import db

class Img(db.Model):
    """Database model to store image metadata and prediction results."""
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    mimetype = db.Column(db.String(50), nullable=False)
    filepath = db.Column(db.String(200), nullable=False)
    prediction = db.Column(db.String(10), nullable=False)

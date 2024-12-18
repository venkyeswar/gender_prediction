from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def db_init(app):
    """Initialize the database and create tables if they don't exist."""
    db.init_app(app)
    with app.app_context():
        db.create_all()

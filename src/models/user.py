#from flask_sqlalchemy import SQLAlchemy

#db = SQLAlchemy()
from . import db  # <-- 添加这行，从 models/__init__.py 导入共享的 db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def __repr__(self):
        return '<User %r>' % self.username

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email
        }


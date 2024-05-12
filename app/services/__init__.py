from celery import Celery
from flask import Flask
import os
from ..model import db

celery_service = Celery('chat_', backend='redis://127.0.0.1:6379/0', broker='redis://127.0.0.1:6379/0')

def create_celery_app():
    celery_app = Flask(__name__, instance_relative_config=True)
    celery_app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("SERVICE_DATABASE_URL")
    celery_app.config['SQLALCHEMY_ECHO'] = True
    celery_app.config['CELERY_RESULT_BACKEND'] = os.environ.get('REDIS_URL')
    db.init_app(celery_app)
    return celery_ap
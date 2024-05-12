import json
import os
from flask import Flask
from flask_cors import CORS
# from flask_script import Manager
from sqlalchemy.orm.attributes import flag_modified
from flask_sqlalchemy import SQLAlchemy
# from flask_jwt_extended import (
#     JWTManager
# )

from .views import chat
from .model import db
# from flask_sieve import Sieve
# from flask_caching import Cache

# cache = Cache(config={
#     'CACHE_TYPE': 'redis',
#     'CACHE_KEY_PREFIX': 'customer_cache',
#     'CACHE_REDIS_HOST': 'localhost',
#     'CACHE_REDIS_PORT': '6379',
#     'CACHE_REDIS_URL': 'redis://localhost:6379'
#     })
def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.app_context().push()
    with app.app_context():
        app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("SERVICE_DATABASE_URL")
        app.register_blueprint(chat)
        # jwt = JWTManager(app)
        db.init_app(app)
        # cache.init_app(app)
        # Sieve(app)
        CORS(app,supports_credentials=True, origins="*")
        return app
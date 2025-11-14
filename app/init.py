from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from celery import Celery
import os

db = SQLAlchemy()
celery = Celery()

def create_app(config_class='config.production.ProductionConfig'):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    db.init_app(app)
    
    # Celery configuration
    celery.conf.update(app.config)
    
    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.camera import camera_bp
    from app.routes.faces import faces_bp
    from app.routes.motion import motion_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(camera_bp, url_prefix='/camera')
    app.register_blueprint(faces_bp, url_prefix='/faces')
    app.register_blueprint(motion_bp, url_prefix='/motion')
    
    # Create storage directories
    from app.utils.recording_manager import create_storage_dirs
    with app.app_context():
        create_storage_dirs()
        db.create_all()
    
    return app

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)
    
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    
    celery.Task = ContextTask
    return celery
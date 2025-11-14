from app import create_app, make_celery

app = create_app()
celery = make_celery(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
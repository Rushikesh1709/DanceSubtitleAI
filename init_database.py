"""Create the SQLite database and the required folders:  python init_database.py"""
from app import create_app
from app.models import ProcessingJob
from app.db import session_scope


def main():
    app = create_app()
    with session_scope(app.extensions["db_factory"]) as s:
        count = s.query(ProcessingJob).count()
    print("Database ready:", app.config["DATABASE_URL"])
    print("Upload folder   :", app.config["UPLOAD_DIR"])
    print("Processed folder:", app.config["PROCESSED_DIR"])
    print(f"Existing processing jobs: {count}")


if __name__ == "__main__":
    main()

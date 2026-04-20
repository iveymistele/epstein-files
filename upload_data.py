import json
import logging
from pymongo import MongoClient
import os

logging.basicConfig(
    filename="upload.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Config
MONGO_PASS = os.getenv('MONGO_PASS')
MONGO_USER = "iveymistele"
MONGO_CLUSTER = "cluster0.93xer.mongodb.net" 

MONGO_URI = f"mongodb+srv://{MONGO_USER}:{MONGO_PASS}@{MONGO_CLUSTER}/?retryWrites=true&w=majority"

DB_NAME = "epstein_db"
COLLECTION_NAME = "emails"
JSON_FILE = "emails_with_metadata.json"

def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for doc in data:
        doc["_id"] = doc["document_id"]
        doc["timestamp"] = f'{doc.get("date", "")}T{doc.get("time", "")}'
        doc["participant_names"] = [p.get("name", "") for p in doc.get("participants", []) if p.get("name")]
        doc["participant_emails"] = [p.get("email", "") for p in doc.get("participants", []) if p.get("email")]
        doc["participant_count"] = len(doc.get("participants", []))
        doc["attachment_count"] = len(doc.get("attachment_names", []))
        doc["url_count"] = len(doc.get("urls", []))
        docs.append(doc)

    if docs:
        collection.insert_many(docs, ordered=False)
        logging.info("Inserted %d documents", len(docs))
        print(f"Inserted {len(docs)} documents into {DB_NAME}.{COLLECTION_NAME}")

if __name__ == "__main__":
    main()
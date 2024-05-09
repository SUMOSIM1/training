from abc import ABC

import pymongo
from bson import ObjectId


def create_client() -> ABC:
    return pymongo.MongoClient("mongodb://localhost:27017/")


def _sim_collection(client: pymongo.MongoClient):
    db = client["sumosim"]
    return db["simulations"]


def insert(client: pymongo.MongoClient, sim_dict: dict) -> str:
    sims = _sim_collection(client)
    result = sims.insert_one(sim_dict)
    str_id = str(result.inserted_id)
    return str_id


def find(client: pymongo.MongoClient, id: str) -> dict:
    raise NotImplementedError()


def find_all(client: pymongo.MongoClient) -> dict:
    sims = _sim_collection(client)
    return list(sims.find())


def update_status(client: pymongo.MongoClient, doc_id: str, status: str, message: str):
    sims = _sim_collection(client)
    update_document = {"$set": {"status": status, "message": message}}
    obj_id = ObjectId(doc_id)
    result = sims.update_one({"_id": obj_id}, update_document)
    print(f"### update {result}")


def find_running(
    client: pymongo.MongoClient, status: str, base_port: int
) -> list[dict]:
    sims = _sim_collection(client)
    query = {
        "status": status,
        "base_port": base_port,
    }
    return list(sims.find(query))

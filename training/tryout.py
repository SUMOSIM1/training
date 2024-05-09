from dataclasses import dataclass
from datetime import datetime

import bson
import pymongo
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Simulation:
    started_at: datetime = datetime.now()  # Default to current time
    status: str = "running"  # Default status
    base_port: int = 8080  # Default base port


def create_simulations():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["sumosim"]

    db.create_collection(
        "simulations",
        check_exists=False,
    )
    print(f"--- collections {db.list_collection_names()}")


def write():
    with pymongo.MongoClient("mongodb://localhost:27017/") as client:
        db = client["sumosim"]
        sims = db["simulations"]
        sim = Simulation(
            started_at=datetime.now(),
            status="running",
            base_port=555,
        )
        sim_dict = sim.to_dict()
        str_id = str(sims.insert_one(sim_dict).inserted_id)
        print(f"---> inserted {str_id}, {sim}")
        object_id = bson.ObjectId(str_id)
        sim1 = sims.find_one({"_id": object_id})
        sim2 = Simulation.from_dict(sim1)
        print(f"-- sim:{sim}")
        print(f"-- sim1(dict):{sim1}")
        print(f"-- sim2:{sim2}")


def main():
    # create_simulations()
    write()

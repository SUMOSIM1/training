import simdb as db


def all_sims():
    with db.create_client() as c:
        for s in db.find_all(c):
            print(s)


def main():
    all_sims()

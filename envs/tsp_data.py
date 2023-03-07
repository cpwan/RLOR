import logging
import pickle

root_dir = "/home/cwan5/OR/attention-learn-to-route/data/tsp/"


def load(filename, root_dir=root_dir):
    return pickle.load(open(root_dir + filename, "rb"))


file_catalog = {
    "test": {
        20: "tsp20_test_seed1234.pkl",
        50: "tsp50_test_seed1234.pkl",
        100: "tsp100_test_seed1234.pkl",
    },
    "eval": {
        20: "tsp20_validation_seed4321.pkl",
        50: "tsp50_validation_seed4321.pkl",
        100: "tsp100_validation_seed4321.pkl",
    },
}


class lazyClass:
    data = {
        "test": {},
        "eval": {},
    }

    def __getitem__(self, index):
        partition, nodes, idx = index
        if not (partition in self.data) or not (nodes in self.data[partition]):
            logging.warning(
                f"Data sepecified by ({partition}, {nodes}) was not initialized. Attepmting to load it for the first time."
            )
            data = load(file_catalog[partition][nodes])
            self.data[partition][nodes] = [tuple(instance) for instance in data]

        return self.data[partition][nodes][idx]


TSPDataset = lazyClass()

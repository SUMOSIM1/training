import training.sgym.qlearn as ql
import numpy as np


def main():
    print("tryout")
    row = np.array([9, 10.6, 10.59999, 5, 10.599, 10.5999])
    for i in range(10):
        x = ql.FetchType.LAZY_S_T5.fetch(row)
        print(f"{i} {x}")

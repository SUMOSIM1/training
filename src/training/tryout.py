import training.helper as hlp
import random as rnd
import pprint as pp


def main():
    n = 80
    n1 = 5
    data = [rnd.random() for _ in range(n)]
    # pp.pprint(data)
    a, b, m = hlp.split_data(data, n1)
    c = [len(x) for x in b]
    pp.pprint(a)
    pp.pprint(b)
    pp.pprint(c)
    pp.pprint(m)

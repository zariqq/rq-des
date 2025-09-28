import numpy as np
import pandas as pd
from multiprocessing import Pool
from sim_cp import sim, run_i_n
import time


def pdf_to_cdf(pdf: np.array):
    cdf = pdf.copy()

    for ax in range(pdf.ndim):
        np.cumsum(cdf, axis=ax, out=cdf)
    return cdf


def equalize_length(a, b):
    len_a, len_b = len(a), len(b)
    max_len = max(len_a, len_b)

    out_a = np.zeros(max_len)
    out_b = np.zeros(max_len)

    out_a[:len_a] = a
    out_b[:len_b] = b

    return out_a, out_b


def kolm_dist(time):
    i1, n1 = run_i_n(time)
    i2, n2 = run_i_n(time)

    print(f"10^{np.log10(time) // 1}")

    i1, i2 = equalize_length(i1, i2)
    n1, n2 = equalize_length(n1, n2)

    i1, i2 = i1 / i1.sum(), i2 / i2.sum()
    n1, n2 = n1 / n1.sum(), n2 / n2.sum()

    f11 = pdf_to_cdf(i1)
    f12 = pdf_to_cdf(n1)

    f21 = pdf_to_cdf(i2)
    f22 = pdf_to_cdf(n2)

    return (
        np.max(np.abs(f11 - f21)),
        np.max(np.abs(f12 - f22)),
        f"10^{np.log10(time) // 1}",
    )


def diff(start=5, end=7, step=1):

    times = [10**i for i in range(start, end, step)]
    table_len = len(times)

    with Pool(end - step) as pool:
        result = pool.map(kolm_dist, times)
        print(*result, sep="\n")

    # return pd.DataFrame({
    #     'D1': res[0],
    #     'D2': res[1]
    # }, index=[f'10^{i}' for i in range(start, end, step)])


if __name__ == "__main__":
    t1 = time.time()
    res = kolm_dist(10**9)
    print(res)
    t2 = time.time()
    print("elapse time: ", (t2 - t1), "sec")

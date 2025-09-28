import numpy as np
from numba import jit, njit, config, threading_layer

config.NUMBA_THREADING_LAYER_PRIORITY = "tbb omp workqueue"


@njit(inline="always")
def set_one(arr, i, dt):
    if i >= len(arr):
        arr.append(0.0)

    arr[i] += dt


@njit(inline="always")
def find_min_index(arr):
    idx = 0
    if arr[1] < arr[idx]:
        idx = 1
    if arr[2] < arr[idx]:
        idx = 2
    if arr[3] < arr[idx]:
        idx = 3
    if arr[4] < arr[idx]:
        idx = 4
    return idx


STATE_STEPS = np.array(
    [
        [
            [  # i = 0, n = 0
                [[0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],  # k = 0
                [[1, 0, 0], [0, 1, -1], [0, 0, -1], [0, 0, 0], [0, 0, 0]],  # k = 1
            ],
            [  # i = 0, n > 0
                [[0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, -1, 1]],
                [[1, 0, 0], [0, 1, -1], [0, 0, -1], [0, 0, 0], [1, -1, 0]],
            ],
        ],
        [
            [  # i > 0, n = 0
                [[0, 0, 1], [0, 0, 0], [0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                [[1, 0, 0], [0, 1, -1], [0, 0, -1], [0, 0, 0], [0, 0, 0]],
            ],
            [  # i > 0, n > 0
                [[0, 0, 1], [0, 0, 0], [0, 0, 0], [-1, 0, 1], [0, -1, 1]],
                [[1, 0, 0], [0, 1, -1], [0, 0, -1], [0, 0, 0], [1, -1, 0]],
            ],
        ],
    ],
    r"int8",
)


@jit(nopython=True, parallel=True)
def random_generator():
    return -np.log(1 - np.random.uniform(0, 1, 100_000_000))


@jit(nopython=True)
def run_i_n(
    sim_time_lim,
    lam=np.float64(0.8),
    gam=np.float64(2.0),
    mu=np.float64(1),
    sig1=np.float64(0.08),
    sig2=np.float64(0.07),
):
    buffer = random_generator()
    current_idx = 0

    t = 0.0
    p_i = []
    p_n = []

    i, n, k = (0, 0, 0)
    dts = np.array([1, 1, 1, 1, 1], r"float64")

    while t < sim_time_lim:
        dts[0] = buffer[current_idx + 0] / lam
        dts[1] = buffer[current_idx + 1] / gam
        dts[2] = buffer[current_idx + 2] / mu
        dts[3] = buffer[current_idx + 3]
        dts[4] = buffer[current_idx + 4]

        current_idx += 5

        ci, cn = (i > 0, n > 0)

        if k == 0:
            dts[1] = dts[2] = np.inf
            dts[3] = dts[3] / (i * sig1) if ci else np.inf
        else:
            dts[3] = np.inf
        dts[4] = dts[4] / (n * sig2) if cn else np.inf

        min_idx = find_min_index(dts)

        if i >= len(p_i):
            p_i.append(0.0)

        if n >= len(p_n):
            p_n.append(0.0)

        p_i[i] += dts[min_idx]
        p_n[n] += dts[min_idx]

        t += dts[min_idx]
        i += STATE_STEPS[ci * 1, cn * 1, k][min_idx][0]
        n += STATE_STEPS[ci * 1, cn * 1, k][min_idx][1]
        k += STATE_STEPS[ci * 1, cn * 1, k][min_idx][2]

        if current_idx + 11 >= 100_000_000:
            buffer = random_generator()
            current_idx = 0

    return p_i, p_n


def sim(
    sim_time_lim,
    it=5,
    lam=np.float64(0.8),
    gam=np.float64(2.0),
    mu=np.float64(1),
    sig1=np.float64(0.08),
    sig2=np.float64(0.07),
):
    experiments = np.zeros((2 * it, 500), "float64")
    for i in range(it):
        pi, pn = run_i_n(sim_time_lim, lam, gam, mu, sig1, sig2)

        for j in range(len(pi)):
            experiments[i * 2][j] += pi[j]
        for j in range(len(pn)):
            experiments[i * 2 + 1][j] += pn[j]

    mn_pi = np.mean(experiments[::2], 0)
    mn_pn = np.mean(experiments[1::2], 0)

    return (mn_pi / mn_pi.sum(), mn_pn / mn_pn.sum())


sim(10**5, 2)

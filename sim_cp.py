import numpy as np
from numba import jit, njit, cuda
import torch
import warnings

warnings.filterwarnings("ignore")


# @njit(inline="always")
def set_one(arr, i, dt):
    if i >= len(arr):
        arr.append(0.0)

    arr[i] += dt


# coefficets of parametrs
@njit(inline="always")
def koefs(tp):
    if tp[0] == 0:
        if tp[1] == 0:  # i = n = 0
            return np.array(
                [[1, np.inf, np.inf, np.inf, np.inf], [1, 1, 1, np.inf, np.inf]],
                "float64",
            )[tp[2]]
        else:  # i = 0, n > 0
            return np.array(
                [[1, np.inf, np.inf, np.inf, 1 / tp[1]], [1, 1, 1, np.inf, 1 / tp[1]]],
                "float64",
            )[tp[2]]
    else:
        if tp[1] == 0:  # i > 0, n = 0
            return np.array(
                [[1, np.inf, np.inf, 1 / tp[0], np.inf], [1, 1, 1, np.inf, np.inf]],
                "float64",
            )[tp[2]]
        else:  # i > 0, n > 0
            return np.array(
                [
                    [1, np.inf, np.inf, 1 / tp[0], 1 / tp[1]],
                    [1, 1, 1, np.inf, 1 / tp[1]],
                ],
                "float64",
            )[tp[2]]


@njit(inline="always")
def get_state_config(lam=0.8, gam=2.0, mu=1, sig1=0.08, sig2=0.07):
    inv_lam = np.inf if lam == 0 else 1.0 / lam
    inv_gam = np.inf if gam == 0 else 1.0 / gam
    inv_mu = np.inf if mu == 0 else 1.0 / mu
    inv_sig1 = np.inf if sig1 == 0 else 1.0 / sig1
    inv_sig2 = np.inf if sig2 == 0 else 1.0 / sig2

    STATE_EDGES = np.array(
        [
            [
                [  # i = 0, n = 0
                    [inv_lam, np.inf, np.inf, np.inf, np.inf],
                    [inv_lam, inv_gam, inv_mu, np.inf, np.inf],
                ],
                [  # i = 0, n > 1
                    [inv_lam, np.inf, np.inf, np.inf, inv_sig2],
                    [inv_lam, inv_gam, inv_mu, np.inf, inv_sig2],
                ],
            ],
            [
                [  # i > 1, n = 0
                    [inv_lam, np.inf, np.inf, inv_sig1, np.inf],
                    [inv_lam, inv_gam, inv_mu, np.inf, np.inf],
                ],
                [  # i > 0, n > 0
                    [inv_lam, np.inf, np.inf, inv_sig1, inv_sig2],
                    [inv_lam, inv_gam, inv_mu, np.inf, inv_sig2],
                ],
            ],
        ],
        "float64",
    )

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
        "int8",
    )

    return STATE_EDGES, STATE_STEPS


def random_generator():
    res = -torch.log(1 - torch.rand(10_000_000, device="mps"))
    return res.cpu().numpy()


def run_i_n(
    sim_time_lim,
    lam=np.float64(0.8),
    gam=np.float64(2.0),
    mu=np.float64(1),
    sig1=np.float64(0.08),
    sig2=np.float64(0.07),
):
    STATE_EDGES, STATE_STEPS = get_state_config(lam, gam, mu, sig1, sig2)

    buffer = random_generator()
    current_idx = 0

    t = 0.0
    p_i = []
    p_n = []

    current_state = np.array([0, 0, 0], "int32")
    dts = np.array([1, 1, 1, 1, 1], "float64")

    print("* * *")
    while t < sim_time_lim:

        edges = STATE_EDGES[
            (current_state[0] > 0) * 1, (current_state[1] > 0) * 1, current_state[2]
        ]

        dts[:] = buffer[current_idx : current_idx + 5] * koefs(current_state) * edges

        current_idx += 5

        min_idx = np.nanargmin(dts)

        set_one(p_i, current_state[0], dts[min_idx])
        set_one(p_n, current_state[1], dts[min_idx])

        t += dts[min_idx]
        current_state += STATE_STEPS[
            (current_state[0] > 0) * 1,
            (current_state[1] > 0) * 1,
            current_state[2],
        ][min_idx]

        if current_idx + 11 >= len(buffer):  # run out buffer
            buffer = random_generator()
            current_idx = 0

    np_p_i = np.array(p_i, "float64")
    np_p_n = np.array(p_n, "float64")

    np.divide(np_p_i, t, np_p_i)
    np.divide(np_p_n, t, np_p_n)

    return np_p_i, np_p_n


run_i_n(10)

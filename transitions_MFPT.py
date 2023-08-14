import numpy as np
import time

def transitions_mfpt(F, B, z0, phi, dt, tmax, N, rho):
    mfpt = 0
    st_initGen = time.time()

    t = 0
    time_steps = 0
    z = (z0 * np.ones((N, 2))).T

    rng = np.random.default_rng()

    while z.shape[1] > 0:
        # Weiner process
        dW = np.sqrt(dt) * rng.standard_normal(size=z.shape)

        # Time incr.
        t = t + dt

        # Euler–Maruyama scheme
        z = z + dt * F(z) + B * dW

        # Count number of steps across all trajectories
        time_steps = time_steps + z.shape[1]

        converged = phi(z) > 1.0 - rho
        z = z[:, ~converged]

        mfpt += t * np.count_nonzero(converged)

    mfpt /= N
    trans_prob = 1 - np.exp(-1 / (mfpt * tmax))

    initGen = time.time() - st_initGen
    #print("Time to generate the {} MFPT trajectories: {}".format(N,initGen))

    return trans_prob, time_steps, mfpt

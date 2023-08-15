import numpy as np

def transitions_direct(F, B, z0, phi, dt, tmax, N, rho):
    tsteps = int(tmax / dt)

    t = 0
    time_steps = 0
    z = (z0 * np.ones((N, 2))).T

    for i in range(tsteps):
        # Weiner processes
        dW = np.sqrt(dt) * np.random.randn(*z.shape)

        # Time incr.
        t += dt

        # Euler–Maruyama scheme
        z = z + dt * F(z) + B * dW

        # Count number of steps across all trajectories
        time_steps += z.shape[1]

        converged = phi(z) > 1 - rho
        z = z[:, ~converged]

    ntrans = N - z.shape[1]
    trans_prob = ntrans / N

    return trans_prob, time_steps

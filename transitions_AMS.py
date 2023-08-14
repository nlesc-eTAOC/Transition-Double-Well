import numpy as np
import random

def transitions_ams(F, B, z0, phi, dt, tmax, N, rho):
    N2 = N * 1
    N3 = 1000000

    # Max number of steps for sufficient increase of react. function
    M = 1000

    experiments = []

    # First pass, storing only experiments going beyond phi(z) = 0.1
    # (i.e. surface C around A)
    for _ in range(N2):
        t = 0
        z = (z0 * np.ones((1, 2))).T
        steps = 0
        converged = False
        while not converged:
            # Weiner processes (all the steps at once)
            dW = np.sqrt(dt) * np.random.randn(len(z), M)

            for j in range(M):
                t += dt
                z = z + dt * F(z) + B * dW[:, [j]]
                dist = phi(z)
                if dist > 0.1:
                    experiment = {
                        "start_time": t,
                        "x": [z],
                        "t": [0],
                        "d": [dist],
                        "return_time": 0,
                        "max_dist": dist,
                        "steps": steps + j
                    }
                    experiments.append(experiment)
                    converged = True
                    break
            steps += M

    # Second pass: restart these experiments, keeping track of where and when the
    # traj. reaches new maximum
    # If traj. goes near B, converged
    # If traj. goes back near A, converged, keep track of time to return to A
    i = 0
    for exp in experiments:
        i +=1
        t = 0
        z = exp["x"][-1]
        steps = 0
        converged = False
        while not converged:
            # Weiner processes (all the steps at once)
            dW = np.sqrt(dt) * np.random.randn(len(z), M)

            for j in range(M):
                t += dt
                z = z + dt * F(z) + B * dW[:, [j]]
                dist = phi(z)
                if dist > exp["max_dist"]:
                    exp["x"].append(z)
                    exp["t"].append(t)
                    exp["d"].append(dist)
                    exp["steps"] = exp["steps"] + steps + j
                    exp["max_dist"] = dist
                    if dist > 1 - rho:
                        converged = True
                        break
                elif dist < rho:
                    if exp["return_time"] == 0:
                        exp["return_time"] = t
                    exp["steps"] = exp["steps"] + steps + j
                    converged = True
                    break
            steps += M

    its = 0
    l = []
    w = [1]
    for _ in range(N3):
        min_val = 1
        min_idx_list = [0]
        # Loop over the first N experiments
        # Getting the index of (all) the experiment(s) having the smallest
        # max_dist
        for j in range(N):
            if experiments[j]["max_dist"] < min_val:
                min_val = experiments[j]["max_dist"]
                min_idx_list = [j]
            elif experiments[j]["max_dist"] == min_val:
                min_idx_list.append(j)

        # All the traj. reached vicinity of B, --> exit
        if min_val > 1 - rho:
            break

        # Get the weight (probability) of the trajectories we kept
        # over the successive iterations of this selection process
        l.append(len(min_idx_list))
        w.append(w[-1] * (1 - l[-1] / N))

        # If all the traj. got to the same min_val, --> exit
        if l[-1] == N:
            break

        # For each trajectory we've identified and discarded, branch
        # one of the others
        for min_idx in min_idx_list:

            # Randomly pick any other trajectories
            idx = min_idx
            while idx == min_idx:
                idx = random.randint(0, N - 1)

            rest_exp = experiments[idx]

            # Take a restart point on the randomly selected traj that has a value 
            # close, but larger than the min_val
            same_dist_idx = 0
            while rest_exp["d"][same_dist_idx] < min_val:
                same_dist_idx += 1

            # Overwrite the discarded trajectory with a new one
            # restarting from the randomly selected one, starting when
            # it get past min_val
            exp = experiments[min_idx]
            exp["x"] = rest_exp["x"][:same_dist_idx+1]
            exp["t"] = rest_exp["t"][:same_dist_idx+1]
            exp["d"] = rest_exp["d"][:same_dist_idx+1]
            exp["max_dist"] = rest_exp["d"][same_dist_idx]
            t = exp["t"][-1]
            z = exp["x"][-1]
            steps = 0
            converged = False
            while not converged:
                # Weiner processes (all the steps at once)
                dW = np.sqrt(dt) * np.random.randn(len(z), M)

                for j in range(M):
                    t += dt
                    z = z + dt * F(z) + B * dW[:, [j]]
                    dist = phi(z)
                    if dist > exp["max_dist"]:
                        exp["x"].append(z)
                        exp["t"].append(t)
                        exp["d"].append(dist)
                        exp["max_dist"] = dist
                        if dist > 1 - rho:
                            exp["steps"] = exp["steps"] + steps + j
                            converged = True
                            break
                    elif dist < rho:
                        exp["steps"] = exp["steps"] + steps + j
                        converged = True
                        break
                steps += M
        its += 1


    total_tr = 0
    total_t1 = 0
    total_t2 = 0
    num_t1 = N2
    num_t2 = 0

    for exp in experiments:
        total_t1 += exp["start_time"]
        total_t2 += exp["return_time"]
        if exp["return_time"] > 0:
            num_t2 += 1

    W = N * w[-1]
    for i in range(its):
        W += l[i] * w[i]

    converged = 0
    time_steps = 0
    for i in range(N):
        if experiments[i]["max_dist"] > 1 - rho:
            converged += 1
            total_tr += experiments[i]["t"][-1]
        time_steps += experiments[i]["steps"]
    alpha = converged * w[-1] / W

    meann = 1.0 / alpha - 1.0
    mfpt = meann * (total_t1 / num_t1 + total_t2 / num_t2) + total_t1 / num_t1 + total_tr / converged

    trans_prob = 1.0 - np.exp(-1.0 / mfpt * tmax)

    # print("Trans. prob: {}".format(trans_prob))

    return trans_prob, time_steps, mfpt

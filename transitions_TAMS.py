import multiprocessing as mp
import time
import numpy as np
import os

def transitions_tams(F, B, z0, phi, dt, tmax, N, N3, rho):

    experiments = []

    # Generate N trajectories up to tmax
    st_initGen = time.time()
    for _ in range(N):
        t = 0.0
        z = (z0 * np.ones((1, 2))).T
        nstep = int(np.ceil(1 / dt * (tmax - t)))

        # Weiner processes (all the steps at once)
        dW = np.sqrt(dt) * np.random.randn(len(z), nstep)

        exp = {'max_dist': 0, 'x': [z], 't': [t], 'd': [0], 'steps': nstep}

        for j in range(nstep):
            t = t + dt
            z = z + dt * F(z) + B * dW[:, [j]]
            dist = phi(z)
            if dist > exp['max_dist']:
                exp['x'].append(z)
                exp['t'].append(t)
                exp['d'].append(dist)
                exp['max_dist'] = dist

        experiments.append(exp)

    initGen = time.time() - st_initGen
    print("Time to generate the initial {} trajectories: {}".format(N,initGen))

    its = 0
    l = []
    w = [1]

    st_fixTraj = time.time()
    for i in range(N3):
        st_ite = time.time()
        min_val = 1
        min_idx_list = [0]

        # Loop over the N experiments
        # Getting the index of (all) the experiment(s) having the smallest
        # max_dist
        for j in range(N):
            if experiments[j]['max_dist'] < min_val:
                min_val = experiments[j]['max_dist']
                min_idx_list = [j]
            elif experiments[j]['max_dist'] == min_val:
                min_idx_list.append(j)

        # All the traj. reached vicinity of B, --> exit
        if min_val > 1 - rho:
            print("All trajs successfully reached B")
            break

        # Get the weight (probability) of the trajectories we kept
        # over the successive iterations of this selection process
        l.append(len(min_idx_list))
        w.append(w[-1] * (1 - l[-1] / N))

        # If all the traj. got to the same min_val, --> exit
        if l[-1] == N:
            print("All trajs stalled -> exiting")
            break

        # For each trajectory we've identified and discarded, branch
        # one of the others
        for min_idx in min_idx_list:

            # Randomly pick any other trajectories
            idx = min_idx
            while idx in min_idx_list:
                idx = np.random.randint(N)

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
            exp['x'] = rest_exp['x'][:same_dist_idx+1]
            exp['t'] = rest_exp['t'][:same_dist_idx+1]
            exp['d'] = rest_exp['d'][:same_dist_idx+1]
            exp['max_dist'] = rest_exp['d'][same_dist_idx]
            t = exp['t'][-1]
            z = exp['x'][-1]
            M = int(np.ceil(1 / dt * (tmax - t)))
            exp['steps'] += M

            # Weiner processes (all the steps at once)
            dW = np.sqrt(dt) * np.random.randn(len(z), M)

            for j in range(M):
                t = t + dt
                z = z + dt * F(z) + B * dW[:, [j]]
                dist = phi(z)
                if dist > exp['max_dist']:
                    exp['x'].append(z)
                    exp['t'].append(t)
                    exp['d'].append(dist)
                    exp['max_dist'] = dist
                    if dist > 1 - rho:
                        exp['steps'] -= M - j
                        break

        its += 1
        iteTime = time.time() - st_ite
        #print("Time for ite {} fixing {} trajs : {}".format(its-1,l[-1],iteTime))

    fixTraj = time.time() - st_fixTraj
    print("Time to fix the trajectories with {} ites: {}".format(its,fixTraj))

    W = N * w[-1]
    for i in range(its):
        W += l[i] * w[i]

    Nb = 0
    time_steps = 0

    # Compute how many traj. converged to the vicinity of B
    for exp in experiments:
        if exp['max_dist'] > 1 - rho:
            Nb += 1
        time_steps += exp['steps']

    trans_prob = Nb * w[-1] / W

    print(trans_prob)
    exit()

    return trans_prob, time_steps

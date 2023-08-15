import multiprocessing as mp
import time
import numpy as np
import copy
import os

# Parallel processes spawned by MP do not have access to global
# variables -> need to redefine this
def dist_loc(x,a,b):
    vA = x - (a * np.ones((x.shape[1], 2))).T
    vB = x - (b * np.ones((x.shape[1], 2))).T
    da = np.sum(vA**2, axis=0)
    db = np.sum(vB**2, axis=0)
    f1 = 0.5
    f2 = 1.0 - f1
    y = f1 - f1 * np.exp(-8 * da) + f2 * np.exp(-8 * db)
    return y

def generate_traj(F, B, z0, dt, nstep, rho):
    t = 0
    z = (z0 * np.ones((1, 2))).T

    # Weiner processes (all the steps at once)
    dW = np.sqrt(dt) * np.random.randn(len(z), nstep)

    exp = {'max_dist': 0, 'x': [z], 't': [t], 'd': [0], 'steps': nstep}

    for j in range(nstep):
        t = t + dt
        z = z + dt * F(z) + B * dW[:, [j]]
        dist = dist_loc(z,z0,([1.0,0.0]))
        if dist > exp['max_dist']:
            exp['x'].append(z)
            exp['t'].append(t)
            exp['d'].append(dist)
            exp['max_dist'] = dist

    return exp


def regenerate_singleExp(ovrw_expe, rst_exp, minVal, F, B, dt, tmax, rho):
    # Take a restart point on the randomly selected traj that has a value 
    # close, but larger than the min_val
    same_dist_idx = 0
    while rst_exp["d"][same_dist_idx] < minVal:
        same_dist_idx += 1

    # Create a new trajectory, copying the first part of the restart one
    exp = {'max_dist': rst_exp['d'][same_dist_idx],
           'x': rst_exp['x'][:same_dist_idx+1],
           't': rst_exp['t'][:same_dist_idx+1],
           'd': rst_exp['d'][:same_dist_idx+1],
           'steps': ovrw_expe['steps']}

    # Regenerate past min_val
    t = exp['t'][-1]
    z = exp['x'][-1]
    nstep = int(np.ceil(1 / dt * (tmax - t)))
    exp['steps'] += nstep

    # Weiner processes (all the steps at once)
    dW = np.sqrt(dt) * np.random.randn(len(z), nstep)

    for j in range(nstep):
        t = t + dt
        z = z + dt * F(z) + B * dW[:, [j]]
        dist = dist_loc(z,([-1.0,0.0]),([1.0,0.0]))
        if dist > exp['max_dist']:
            exp['x'].append(z)
            exp['t'].append(t)
            exp['d'].append(dist)
            exp['max_dist'] = dist
            if dist > 1 - rho:
                exp['steps'] -= nstep - j
                break

    return exp


def transitions_ptams(F, B, z0, phi, dt, tmax, N, N3, rho):
    # Multiproc
    Nproc = 12

    # Generate N trajectories up to tmax
    st_initGen = time.time()
    with mp.Pool(Nproc) as pool:
        expes = []
        for _ in range(N):
            t = 0.0
            nstep = int(np.ceil(1 / dt * (tmax - t)))
            expes.append(pool.apply_async(generate_traj,args=(F,B,z0,dt,nstep,rho)))

        pool.close()
        pool.join()

    # expes is mutable --> need to immutabilize
    experiments = []
    for exp in expes:
        experiments.append(exp.get())

    initGen = time.time() - st_initGen

    its = 0
    l = []
    w = [1]

    pool = mp.Pool(Nproc)

    st_fixTraj = time.time()
    for i in range(int(N3/Nproc)):
        st_ite = time.time()

        # Find the n trajectories with the smallest values of max_dist
        # n being the number of processes
        maxes = np.zeros(len(experiments))
        for i in range(len(experiments)):
            maxes[i] = experiments[i]['max_dist']
        min_idx_list = np.argpartition(maxes, Nproc)[:Nproc]
        min_vals = maxes[min_idx_list]

        # Randomy pick the Nproc experiments we'll restart from
        rest_experiments = []
        for i in range(Nproc):
            rest_idx = min_idx_list[i]
            while rest_idx in min_idx_list:
                rest_idx = np.random.randint(len(experiments))
            rest_experiments.append(copy.deepcopy(experiments[rest_idx]))

        # All the traj. reached vicinity of B, --> exit
        if np.amin(min_vals) > 1 - rho:
            print("All trajs successfully reached B")
            break

        # Get the weight (probability) of the trajectories we kept
        # over the successive iterations of this selection process
        l.append(len(min_idx_list))
        w.append(w[-1] * (1 - l[-1] / N))

        # If all the traj. got to the same min_val, --> exit
        if (np.amax(maxes) - np.amin(maxes)) < 1e-10:
            print("All trajs stalled -> exiting")
            break

        # For each trajectory we've identified and discarded, branch
        # one of the others
        reg_expes = []
        for exp_idx in range(len(min_idx_list)):
            reg_expes.append(pool.apply_async(regenerate_singleExp,args=(experiments[min_idx_list[exp_idx]],
                                                                         rest_experiments[exp_idx],
                                                                         min_vals[exp_idx], F, B, dt, tmax, rho)))

        # Update experiments
        for i in range(len(min_idx_list)):
            experiments[min_idx_list[i]] = copy.deepcopy(reg_expes[i].get())

        its += 1
        iteTime = time.time() - st_ite
        #print("Time for ite {} fixing {} trajs : {}".format(its-1,l[-1],iteTime))


    pool.close()
    fixTraj = time.time() - st_fixTraj
    #print("Time to fix the trajectories with {} ites: {}".format(its,fixTraj))

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

    return trans_prob, time_steps

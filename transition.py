import numpy as np

from transitions_Direct import transitions_direct
from transitions_MFPT import transitions_mfpt
from transitions_AMS import transitions_ams
from transitions_TAMS import transitions_tams
from transitions_TAMSparallel import transitions_ptams


# Define functions F, phi, V, Vx, Vxx

# F = -\nabla V(x)

def F(x):
    return (np.array([x[0] - x[0]**3, -2*x[1]]))


def phi(x):
    return dist_fun(x, zA, zB)


def V(x):
    return (1/4)*x[0]**4 - (1/2)*x[0]**2 + x[1]**2


def Vx(x):
    return np.array([x[0]**3 - x[0], 2*x[1]])


def Vxx(x):
    return np.array([[3*x[0]**2 - 1, 0], [0, 2]])


def dist_fun(x, a, b):
    vA = x - (a * np.ones((x.shape[1], 2))).T
    vB = x - (b * np.ones((x.shape[1], 2))).T
    da = np.sum(vA**2, axis=0)
    db = np.sum(vB**2, axis=0)
    f1 = 0.5
    f2 = 1.0 - f1
    y = f1 - f1 * np.exp(-8 * da) + f2 * np.exp(-8 * db)
    return y


def compute_median(d):
    if len(d) % 2 == 0:
        out = (d[int(len(d) / 2)] + d[int(len(d) / 2) + 1]) / 2
    else:
        out = d[int(np.ceil(len(d) / 2))]
    return out


def make_samples(f, nargout, nsamples, *args):
    data = {}
    data['tp'] = []
    data['steps'] = []
    data['mfpt'] = []
    data['sigma'] = 0
    data['mu'] = 0
    data['median'] = 0
    data['avg_steps'] = 0
    data['Q1'] = 0
    data['Q3'] = 0
    varargout = [0]
    if nargout == 3:
        varargout.append(0)
    for i in range(nsamples):
        if nargout == 3:
            a, b, c = f(*args)
            data['tp'].append(a)
            data['steps'].append(b)
            data['mfpt'].append(c)
            data['avg_steps'] += b / nsamples
            data['mu'] += c / nsamples
            varargout[0] += a / nsamples
            varargout[1] = data['mu']
        else:
            a, b = f(*args)
            data['tp'].append(a)
            data['steps'].append(b)
            data['avg_steps'] += b / nsamples
            data['mu'] += a / nsamples
            varargout[0] = data['mu']
    if nsamples > 1:
        if nargout == 3:
            d = np.sort(data['mfpt'])
        else:
            d = np.sort(data['tp'])
        data['sigma'] = np.sqrt(1 / (nsamples - 1)
                                * np.sum((d - data['mu'])**2))
        data['median'] = compute_median(d)
        data['Q1'] = compute_median(d[:int(len(d) / 2)])
        data['Q3'] = compute_median(d[round(len(d) / 2):])
        if data['mu'] > 0.0:
            data['normalized_error'] = data['sigma'] \
                / data['mu'] * np.sqrt(data['avg_steps'])
        else:
            data['normalized_error'] = 0.0
    return data, *varargout


if __name__ == '__main__':

    # General parameters
    dt = 0.01
    rho = 0.05
    sigma = np.sqrt(0.1)
    #sigma = 0.4
    B = sigma
    Trange = range(2, 10)
    Brange = [B]
    
    z0 = np.array([-1.0, 0.0])
    zA = z0.copy()
    zB = np.array([1.0, 0.0])
    zC = np.array([0.0, 0.0])
    
    samples = 10
    
    Nmfpt = 200
    Ndirect = 200
    Ntams = 1000

    # Generic part
    VxxEv = np.linalg.eigvals(Vxx(zC))
    
    trans_prob_list1 = {}
    trans_prob_list2 = {}
    trans_prob_list3 = {}
    trans_prob_list4 = {}
    trans_prob_list5 = {}
    trans_prob_list6 = {}
    
    mfpt_list1 = []
    mfpt_list3 = []
    mfpt_list5 = []
    
    data_list1 = []
    data_list2 = {}
    data_list3 = []
    data_list4 = {}
    data_list5 = []
    data_list6 = {}
    
    error_list1 = {}
    error_list2 = {}
    error_list3 = {}
    error_list4 = {}
    error_list5 = {}
    error_list6 = {}
    
    normalized_error_list1 = {}
    normalized_error_list2 = {}
    normalized_error_list3 = {}
    normalized_error_list4 = {}
    normalized_error_list5 = {}
    normalized_error_list6 = {}
    
    for Bi in range(len(Brange)):
        B = Brange[Bi]
    
        trans_prob_list1[Bi] = []
        trans_prob_list2[Bi] = []
        trans_prob_list3[Bi] = []
        trans_prob_list4[Bi] = []
        trans_prob_list5[Bi] = []
        trans_prob_list6[Bi] = []
    
        data_list2[Bi] = []
        data_list4[Bi] = []
        data_list6[Bi] = []
    
        error_list1[Bi] = {}
        error_list2[Bi] = {}
        error_list3[Bi] = {}
        error_list4[Bi] = {}
        error_list5[Bi] = {}
        error_list6[Bi] = {}
    
        normalized_error_list1[Bi] = []
        normalized_error_list2[Bi] = []
        normalized_error_list3[Bi] = []
        normalized_error_list4[Bi] = []
        normalized_error_list5[Bi] = []
        normalized_error_list6[Bi] = []
    
        # Theoretical MFPT using Eyring–Kramers Formula
        mfptt = 2 * np.pi / -min(VxxEv) * \
            np.sqrt(abs(np.linalg.det(Vxx(zC))) /
                    np.linalg.det(Vxx(zA))) * np.exp((V(zC) - V(zA)) /
                                                     (sigma**2/2))
        mfpt_list1.append(mfptt)
        data_list1.append(0)
    
        print("Compute MFPT directly, might take a while ...")
        mfpt = 0
        data = 0
        if mfptt < 1e5:
            data, trans_prob, mfpt = make_samples(
                transitions_mfpt, 3, samples, F, B, z0, phi, dt, 1, Nmfpt, rho)
        mfpt_list3.append(mfpt)
        data_list3.append(data)
    
        print("Compute MFPT with AMS, might take a while ...")
        data, trans_prob, mfpt = make_samples(
             transitions_ams, 3, samples, F, B, z0, phi, dt, 1, Nmfpt, rho)
        mfpt_list5.append(mfpt)
        data_list5.append(data)
    
        for tmax in Trange:
            print('\n T = {}'.format(tmax))
    
            # Theoretical value
            trans_prob_list1[Bi].append(1 - np.exp(-1 / mfpt_list1[Bi] * tmax))
            print(" => Theor. trans. proba.: {}".format(trans_prob_list1[Bi][-1]))

            trans_prob = 0.0
    
            # Direct Monte-Carlo
            print(" => Compute using direct method")
            data, trans_prob = make_samples(
                transitions_direct, 2, samples, F, B, z0, phi, dt, tmax, Ndirect, rho)
            trans_prob_list2[Bi].append(trans_prob)
            data_list2[Bi].append(data)
            error_list2[Bi][tmax] = [trans_prob - data['Q1'], data['Q3'] - trans_prob]
            normalized_error_list2[Bi].append(np.sqrt(dt) * data['normalized_error'])
            print("    Direct trans. proba.: {}".format(trans_prob))
    
            # Using the direct MFPT computed above
            print(" => Compute using direct MFPT method")
            trans_prob = 1 - np.exp(-1 / mfpt_list3[Bi] * tmax)
            data = data_list3[Bi]
            trans_prob_list3[Bi].append(trans_prob)
            error_list3[Bi][tmax] = [trans_prob - data['Q1'] / data['mu'] * trans_prob,
                                     data['Q3'] / data['mu'] * trans_prob - trans_prob]
            normalized_error_list3[Bi].append(np.sqrt(dt) *
                                              data['normalized_error'] /
                                              data['mu'] * trans_prob)
            print("    Direct MFPT trans. proba.: {}".format(trans_prob))
    
            # Using the AMS MFPT computed above
            print(" => Compute using the MFPT obtained with the AMS method")
            trans_prob = 1 - np.exp(-1 / mfpt_list5[Bi] * tmax)
            data = data_list5[Bi]
            trans_prob_list5[Bi].append(trans_prob)
            error_list5[Bi][tmax] = [trans_prob - data['Q1'] / data['mu'] * trans_prob,
                                     data['Q3'] / data['mu'] * trans_prob - trans_prob]
            normalized_error_list5[Bi].append(np.sqrt(dt) *
                                              data['normalized_error'] /
                                              data['mu'] * trans_prob)
            print("    AMS trans. proba.: {}".format(trans_prob))
    
            # Using the TAMS
            print(" => Compute using direct TAMS method")
            data, trans_prob = make_samples(
                transitions_ptams, 2, samples, F, B, z0, phi, dt,
                tmax, Nmfpt, Ntams, rho)
            trans_prob_list6[Bi].append(trans_prob)
            data_list6[Bi].append(data)
            error_list6[Bi][tmax] = [trans_prob - data['Q1'], data['Q3'] - trans_prob]
            normalized_error_list6[Bi].append(np.sqrt(dt) *
                                              data['normalized_error'])
            print("    TAMS trans. proba.: {}".format(trans_prob))
    
    # Plots
    # TODO

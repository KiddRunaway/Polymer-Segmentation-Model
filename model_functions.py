import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import playsound
import time

def Q(u, x, r):
    u += 1
    x += 1
    sigma = u * r
    exponent = -0.5 * ((x - u/2) / sigma)**2
    factor = 1 / (np.sqrt(2 * np.pi) * sigma)
    Q = factor * np.exp(exponent)
    return Q

def p(x, s):
    x += 1
    N_x = 0
    x, s = np.float64(x), np.float64(s)
    p = (x)**s
    return p

def matrix(N, t, N_x, A, r, s):
    indexes = np.arange(0, N_x, 1)
    B = np.zeros((N_x, N_x))

    for i in range(1, N_x-1):
        for j in range(N_x):
            if j > i:
                B[i, j] = 2 * A * p(j, s) * Q(j, i, r)
            elif j == i:
                B[i, j] = N + t - A * p(j, s)

    B[0, 0] = N + t
    for j in range(1, N_x):
        B[0, j] = 2 * A * p(j, s) * Q(j, 0, r)
    B[-1, -1] = N + t - A * p(N_x, s)
    return B

def model(f_0, N_x, N_t, r, s, N=20):
    x_array = np.arange(1, N_x+1, 1)
    t_array = np.arange(1, N_t+1, 1)

    f_space = np.zeros((N_x, N_t+1))
    f_space[:, 0] = f_0

    for i in t_array:
        A = 1 / np.sum(p(x_array, s) * f_0)
        B = matrix(N, i, N_x, A, r, s)
        f_1 = np.dot(B, f_0) / (N + i + 1)
        f_1 = f_1 / np.sum(f_1)
        f_space[:, i] = f_1
        f_0 = f_1

    return f_space

def contour_plotter(sol, x_array, t_array, dim=2):
    start = time.time() 
    if dim == 2:
        fig, ax = plt.subplots()
    elif dim == 3:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    contour = ax.contourf(t_array, x_array, sol, cmap='viridis', levels=1000)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    cbar = fig.colorbar(contour)
    end = time.time()
    print("countour_plotter: The whole solution took %.3f seconds to plot" % (end-start))
    playsound.playsound('/home/serenity/Documents/コドだ/Period 5 & 6/BEEP.wav')
    plt.show()
    return

def fitness_function(params, f_0, N_x, N_t, f_exp):
        r, s = params
        print(r, s)
        f_model = model(f_0, N_x, N_t, r, s)[:, -1]

        var_m, var_e = np.var(f_model), np.var(f_exp)
        mean_m, mean_e = np.mean(f_model), np.mean(f_exp)
        skew_m, skew_e = sp.stats.skew(f_model), sp.stats.skew(f_exp)
        kurt_m, kurt_e = sp.stats.kurtosis(f_model), sp.stats.kurtosis(f_exp)

        A1, A2, A3, A4 = 2.5, 1, 1, 1
        var = A1 * np.abs((var_e - var_m) / var_e)
        mean = A2 * np.abs((mean_e - mean_m) / mean_e)
        skew = A3 * np.abs((skew_e - skew_m) / skew_e)
        kurtosis =  A4 * np.abs((kurt_e - kurt_m)/ kurt_e)
        total_error = var + mean + skew + kurtosis

        return total_error

def parameter_fitter(f_0, N_x, N_t, f_exp):
    guess = [0.09, 3]
    parameters = sp.optimize.minimize(fitness_function, x0=guess, args=(f_0, N_x, N_t, f_exp), method="BFGS")
    return parameters
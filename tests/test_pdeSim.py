from pdeSim import PoissonFlow2D, PoissonFlow2DCN
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 3

def test_PoissonFlow2D_conv(cg = False):
    '''
    This test function the order of convergence of the 2D Poisson solver using by using the method of manufactured
    solutions. A convergence order of 2 is expected.
    '''
    print('Running test_PoissonFlow2D_conv')
    # The imports

    params = {
        'T': 0.2,
        'nt': 11,
        'nx': [21,41,81],
        'freq': 0,
        'neu_top': False,
        'cg': cg,
        'save_file':True
    }

    filename = 'test_PoissonFlow2D_conv.png'

    e2s = []
    for nx in params['nx']:
        params['ny'] = nx
        dx = 1/(nx-1)
        dy = 1 /(params['ny'] - 1)

        dys = [dy for i in range(params['ny'])]

        x_vals = np.linspace(0, (nx - 1) * dx, nx)
        y_vals = np.array([sum(dys[0:i]) for i in range(len(dys))])

        Y, X = np.meshgrid(y_vals, x_vals, indexing='xy')

        q = 2 * np.pi ** 2 * np.sin(X * np.pi) * np.sin(Y * np.pi)

        # The input arrays
        kappa = np.ones((nx, params['ny']))
        rho = np.ones((nx, params['ny']))
        c = np.ones((nx, params['ny']))
        v = np.zeros((nx, params['ny']))

        # The analytic solution
        u_a = np.sin(X * np.pi) * np.sin(Y * np.pi)

        # The boundary value solver
        b = PoissonFlow2D(q, kappa, v, dx, dys, c, rho, params['freq'], params['neu_top'],
                          cg = params['cg'])
        b.make_matrix()
        u_bv = b.solve()

        # The errors
        e2 = np.sqrt(np.sum((u_a.real - u_bv.real)**2)/nx/params['ny'])
        e2s.append(e2)
        print('Poisson equation solved for a',str(nx), 'x',str(nx),'grid with normalised global error {:0.6e}'.format(e2))

    deltas = [1 / (nx - 1) for nx in params['nx']]
    # A figure
    fig, ax = plt.subplots(1, 1, figsize=[5, 4])
    ax.loglog(deltas, e2s, '.',color='b',alpha=1, markersize = 12, label='e2')
    ax.axis('equal')
    #ax.plot(y_vals, u.real[21, :].T, '.', markersize=12, color='r', alpha=1, label='BV')

    ax.set_xlabel('dx, dy')
    ax.set_ylabel('e2')

    ax.legend(frameon=False)

    fig.tight_layout()
    #ax.set_xlim([0, 1])
    #ax.set_xlim([0, 1])
    # ax.set_ylim([0,1.1])
    if params['save_file']:
        fig.savefig(filename, dpi=300)

    # Find the order of convergence
    def power_law(delta,A,p):
        return A*delta**p



    popt, pcov = curve_fit(power_law, deltas,e2s,p0 = [1,2])
    print('The determined order of convergence to the manufactured solution is {:0.3f}. A value of 2 is expected.'.format(popt[1]))

    assert abs(popt[1] - 2) < 0.1, "Convergence error: order is not 2"
    print('Test passed')

def test_PoissonFlow2D_var_kappa():
    '''
    Test to verify the numerical solution of the Poisson equation in case there is a variable thermal conductivity (kappa). Here,
    a method of manufactured solutions approach is taken using the u_a = np.sin(X * np.pi) * np.sin(Y * np.pi) solution over
    a square domain bounded by (0,0) and (1,1). A variable kappa of np.cos(X * np.pi / 4) * np.cos(Y * np.pi / 4) is used. Some variation
    occurs across the domain but not enough to make kappa = 0 or go negative!

    '''
    print('Running test_PoissonFlow2D_var_kappa')

    params = {
        'nx': 41,
        'ny': 41,
        'freq': 0,
        'neu_top': False,
        'save_file':True
    }

    filename = 'test_PoissonFlow2D_var_kappa.png'

    dx = 1 / (params['nx'] - 1)
    dy = 1 / (params['ny'] - 1)

    dys = [dy for i in range(params['ny'])]

    x_vals = np.linspace(0, (params['nx'] - 1) * dx, params['nx'])
    y_vals = [0] + [sum(dys[:i]) / 2 + sum(dys[1:i - 1]) / 2 for i in range(2, params['ny'] + 1)]

    Y, X = np.meshgrid(y_vals, x_vals, indexing='xy')

    q = np.pi ** 2 * (np.sin(np.pi * Y) * np.cos(np.pi * Y / 4) * (np.sin(np.pi * X) * np.cos(np.pi * X / 4) + 1 / 4 * np.cos(np.pi * X) * np.sin(np.pi * X / 4)) +
                      np.sin(np.pi * X) * np.cos(np.pi * X / 4) * (np.sin(np.pi * Y) * np.cos(np.pi * Y / 4) + 1 / 4 * np.cos(np.pi * Y) * np.sin(np.pi * Y / 4)))

    # The input arrays
    kappa = np.cos(X * np.pi / 4) * np.cos(Y * np.pi / 4)
    rho = np.ones((params['nx'], params['ny']))
    c = np.ones((params['nx'], params['ny']))
    v = np.zeros((params['nx'], params['ny']))

    # The analytic solution
    u_a = np.sin(X * np.pi) * np.sin(Y * np.pi)

    # The boundary value solver with the variable kappa
    a = PoissonFlow2D(q, kappa, v, dx, dys, c, rho, params['freq'], params['neu_top'])
    a.make_matrix()
    u = a.solve()

    # A figure
    fig, ax = plt.subplots(1, 1, figsize=[5, 4])
    ax.plot(y_vals, u_a.real[21, :].T, color='b', alpha=1, label='ana')
    ax.plot(y_vals, u.real[21, :].T, '.', markersize=12, color='r', alpha=1, label='BV')

    ax.set_xlabel('y')
    ax.set_ylabel('u')

    ax.legend(frameon=False)

    fig.tight_layout()
    ax.set_xlim([0, 1])
    ax.set_xlim([0, 1])
    # ax.set_ylim([0,1.1])
    if params['save_file']:
        fig.savefig(filename, dpi=300)

    # The errors
    e2 = np.sqrt(np.sum((u_a.real - u.real) ** 2) / params['nx'] / params['ny'])
    print('Poisson equation solved (BV) for a', str(params['nx']), 'x', str(params['ny']),
          'grid with normalised global error {:0.6e} and a varying thermal conductivity'.format(e2))

    assert e2 < 0.01, 'Assertion error'

    print('Test passed')
    
def test_PoissonFlow2D_var_dy():
    '''
    Uses the method of manufactured solutions to test the Poisson solver in the case of a variable dy.
    '''
    print('Running test_PoissonFlow2D_var_dy')

    params = {
        'dx': 0.025,
        'nx': 41,
        'ny': 41,
        'freq': 0,
        'neu_top': False,
        'save_file':True
    }

    filename = 'test_PoissonFlow2D_var_dy.png'

    params['dys'] = [0.04 for i in range(10)] + [0.62 / 30.5 for i in range(31)]
    #print(params['dys'], sum(params['dys']), len(params['dys']))

    x_vals = np.linspace(0, 1, params['nx'])
    # y_vals = np.array([sum(params['dys'][0:i]) for i in range(len(params['dys']))])

    y_vals = [0] + [sum(params['dys'][:i]) / 2 + sum(params['dys'][1:i - 1]) / 2 for i in range(2, params['ny'] + 1)]

    # The heat source
    Y, X = np.meshgrid(y_vals, x_vals, indexing='xy')
    q = 2 * np.pi ** 2 * np.sin(X * np.pi) * np.sin(Y * np.pi)

    # The input arrays
    kappa = np.ones((params['nx'], params['ny']))
    rho = np.ones((params['nx'], params['ny']))
    c = np.ones((params['nx'], params['ny']))
    v = np.zeros((params['nx'], params['ny']))

    # The analytic solution
    u_a = np.sin(X * np.pi) * np.sin(Y * np.pi)

    # The boundary value solver
    b = PoissonFlow2D(q, kappa, v, params['dx'], params['dys'], c, rho, params['freq'], params['neu_top'])
    b.make_matrix()
    u = b.solve()

    # A figure
    fig, ax = plt.subplots(1, 1, figsize=[5, 4])
    ax.plot(y_vals, u_a.real[21, :].T, color='b', alpha=1, label='ana')
    ax.plot(y_vals, u.real[21, :].T, '.', markersize=12, color='r', alpha=1, label='BV')

    ax.set_xlabel('y')
    ax.set_ylabel('u')

    ax.legend(frameon=False)

    fig.tight_layout()
    ax.set_xlim([0, 1])
    ax.set_xlim([0, 1])
    # ax.set_ylim([0,1.1])
    if params['save_file']:
        fig.savefig(filename, dpi=300)

    # Compare the errors
    e2 = np.sqrt(np.sum((u_a.real - u.real) ** 2) / params['nx'] / params['ny'])
    print('Poisson equation solved (BV) for a', str(params['nx']), 'x', str(params['ny']),
          'grid with normalised global error {:0.6e} and a varying dy'.format(e2))

    assert e2 < 0.01, 'Assertion error'
    print('Test passed')

def test_PoissonFlow2DCN_conv():
    '''
    Test to see if the CN solver converges to the same error (from the manufactured solution) as the boundary value solver.
    '''
    print('Running test_PoissonFlow2DCN_conv')
    params = {
        'T': 5,
        'nt': 250,
        'nx': 41,
        'ny': 41,
        'freq': 0,
        'neu_top': False,
        'cg': False,
        'save_file':True
    }

    filename = 'test_PoissonFlow2DCN_conv.png'

    dx = 1/(params['nx'] - 1 )
    dy = 1 /(params['ny'] - 1)
    dt = params['T'] / (params['nt'] - 1)

    dys = [dy for i in range(params['ny'])]

    x_vals = np.linspace(0, (params['nx'] - 1) * dx, params['nx'])
    y_vals = np.array([sum(dys[0:i]) for i in range(len(dys))])

    Y, X = np.meshgrid(y_vals, x_vals, indexing='xy')

    q = 2 * np.pi ** 2 * np.sin(X * np.pi) * np.sin(Y * np.pi)

    # The input arrays
    kappa = np.ones((params['nx'], params['ny']))
    rho = np.ones((params['nx'], params['ny']))
    c = np.ones((params['nx'], params['ny']))
    v = np.zeros((params['nx'], params['ny']))

    qs = [q for k in range(params['nt'])]
    u0 = np.zeros((params['nx'], params['ny']))

    # The analytic solution
    u_a = np.sin(X * np.pi) * np.sin(Y * np.pi)

    # qs, kappa, rho, c, v, freq, u0, T, nt, dx, dy, nx, ny)
    # The Crank-Nicolson solver
    a = PoissonFlow2DCN(qs, kappa, rho, c, v, u0, params['T'],
                        params['nt'], dx, dys, params['nx'], params['ny'], params['neu_top'])

    a.make_matricies()
    us = a.cn_solve()

    # The boundary value solver
    b = PoissonFlow2D(q, kappa, v, dx, dys, c, rho, params['freq'], params['neu_top'],
                      cg=params['cg'])
    b.make_matrix()
    u_bv = b.solve()

    # A figure
    fig, ax = plt.subplots(1, 1, figsize=[5, 4])

    for i in range(0,params['nt']):
        ax.plot(y_vals, us[i][21, :].T, markersize=12, color='r', alpha=0.5)
    ax.plot(y_vals, u_a.real[21, :].T, color='b', alpha=1, label='ana')

    ax.set_xlabel('y')
    ax.set_ylabel('u')

    ax.legend(frameon=False)

    fig.tight_layout()
    ax.set_xlim([0, 1])
    ax.set_xlim([0, 1])
    # ax.set_ylim([0,1.1])
    if params['save_file']:
        fig.savefig(filename, dpi=300)

    e2s = []
    for u in us:
        e2s.append(np.sqrt(np.sum((u_a.real - u.real)**2)/params['nx']/params['ny']))
    print('Poisson equation solved (CN) for a', str(params['nx']), 'x', str(params['ny']),'grid with normalised global error {:0.6e}'.format(e2s[-1]))
    e2 = np.sqrt(np.sum((u_a.real - u.real) ** 2) / params['nx'] / params['ny'])
    print('Poisson equation solved (BV) for a',str(params['nx']), 'x',str(params['ny']),'grid with normalised global error {:0.6e}'.format(e2))
    l2 = np.sqrt((e2-e2s[-1])**2/e2**2)
    assert l2 < 0.01, 'Assertion error: not converging to same solution as boundary value solver'
    print('Test passed')



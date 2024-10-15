# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:19:49 2019

@author: andrew.ferguson
"""
import numpy as np
import csv

import os
cwd = os.getcwd()
print('cwd: ', cwd)

from thermal_parameters import build_thermal_dictionaries


# import the table of thermal parameters and populate dictionaries that we will use to define the properties
# of each layer

k_dict, c_dict, rho_dict = build_thermal_dictionaries()

def heater_continuous(c_out, rho_out, k_out, q_out, xc, c_dict, rho_dict, k_dict,
                      n_bound, power, vias, heater_metal,dx,dys):
    '''
    Updates the thermal (c, rho, k, Q) arrays, adding a heater, heat-spreader and vias
    at position xc. The word 'continuous' in the name refers to the fact that the heat source is not
    patterned.

    Parameters
    ----------
    c_out np.ndarray: output specific heat capacity array
    rho_out np.ndarray: output density array
    k_out np.ndarray: output thermal conductivity array
    q_out np.ndarray: output heat array
    c_dict dict: specific heat capacities of different materials from file
    rho_dict dict: densities of different materials from file
    k_dict dict: thermal conductivities of different materials from file
    n_bound list of int: the materials stack in the y-direction indexed by quad
    power float: the power/unit length for each quad in the simulation
    vias boolean: include the vias in the stack
    heater_metal boolean: include the heater metal in the stack?

    Returns
    ----------
    tuple containing (c_out, rho_out, k_out, q_out) - these are the input arrays, but updated

    '''

    # the heat spreader
    i = 5
    c_out[xc-10:xc +10, n_bound[i]:n_bound[i + 1]] = c_dict['gold']
    rho_out[xc-10:xc+10, n_bound[i]:n_bound[i + 1]] = rho_dict['gold']
    k_out[xc-10:xc+10, n_bound[i]:n_bound[i + 1]] = k_dict['gold']
    #print(n_bound[i])

    # the vias
    # if vias:
    #     i = 3
    #     c_out[xc+24:xc+32, n_bound[i]:n_bound[i + 2]] = c_dict['gold']
    #     rho_out[xc+24:xc+32, n_bound[i]:n_bound[i + 2]] = rho_dict['gold']
    #     k_out[xc+24:xc+32, n_bound[i]:n_bound[i + 2]] = k_dict['gold']
    #
    #     c_out[xc-32:xc-24, n_bound[i]:n_bound[i + 2]] = c_dict['gold']
    #     rho_out[xc-32:xc-24, n_bound[i]:n_bound[i + 2]] = rho_dict['gold']
    #     k_out[xc-32:xc-24, n_bound[i]:n_bound[i + 2]] = k_dict['gold']
    #
    #     c_out[xc+20:xc+36, n_bound[i + 2]:n_bound[i + 4]] = c_dict['gold']
    #     rho_out[xc+20:xc+36, n_bound[i + 2]:n_bound[i + 4]] = rho_dict['gold']
    #     k_out[xc+20:xc+36, n_bound[i + 2]:n_bound[i + 4]] = k_dict['gold']
    #
    #     c_out[xc-36:xc-20, n_bound[i + 2]:n_bound[i + 4]] = c_dict['gold']
    #     rho_out[xc-36:xc-20, n_bound[i + 2]:n_bound[i + 4]] = rho_dict['gold']
    #     k_out[xc-36:xc-20, n_bound[i + 2]:n_bound[i + 4]] = k_dict['gold']

    # put in the heater
    if heater_metal:
        i = 3
        c_out[xc - 10:xc + 10, n_bound[i]:n_bound[i + 1]] = c_dict['platinum']
        rho_out[xc - 10:xc + 10, n_bound[i]:n_bound[i + 1]] = rho_dict['platinum']
        k_out[xc - 10:xc + 10, n_bound[i]:n_bound[i + 1]] = k_dict['platinum']

    # put the heat source in
    i = 3
    q_out[xc - 9:xc + 10, n_bound[i]:n_bound[i + 1]] = power/19/dx/dys[i]
    #print('n_bound_heater',n_bound[i],dys[i])
    return (c_out, rho_out, k_out, q_out)


def model(wet = True, vias = True, power = 1, heater_metal = True, v_max = 0):
    '''
    Model of the polyimide chip.

    Notes
    1. The 1 um Ti + Ag back side metallisation is not included. It will have around the same thermal parameters as
    Si and is only 1 um thick by comparision with the 725 um of the Si wafer. Conclusion. Negligable.
    2. The 4 um patterned Al layer on the top of the silicon wafer is not included. Again it has around the
    same thermal parameters as Si and is relatively thin by comparision with the Si wafer.
    3. The ruthenium layer is included, but only in the region of the heater where the areal fill factor is greatest.

    Parameters
    ----------
    wet boolean: is there water or air above the chip?
    vias boolean: are the through polyimide vias included
    power float: the power/unit length for each quad in the simulation
    heater_metal boolean: is the ruthenium heater metal included in the simulation?
    v_max float: the velocity in the centre of the flow profile, i.e. the maximum

    Returns
    ----------
    The function returns the following tuple (k_out,c_out,rho_out,q_out,v_out,x,y,dx,dy,nx,ny)


    '''
    # user defined parameters
    # the materials stack from top to bottom including materials, thicknesses and number of cells in the
    # vertical direction in each layer
    layers = []
    layers.append({'material': 'silicon_dioxide', 'thick': 1000e-6, 'n': 10})
    if wet:
        layers.append({'material': 'water', 'thick': 28e-6, 'n': 14})
        layers.append({'material': 'water', 'thick': 2e-6, 'n': 1})
    else:
        layers.append({'material': 'air', 'thick': 28e-6, 'n': 14})
        layers.append({'material': 'air', 'thick': 2e-6, 'n': 1})
    layers.append({'material':'silicon_nitride', 'thick': 8.5e-7, 'n': 1})
    layers.append({'material':'silicon_nitride','thick':1.5e-7,'n':1})
    layers.append({'material':'silicon_dioxide','thick':500e-6,'n':1})
    #layers.append({'material':'polyimide','thick':10e-6,'n':10})
    #layers.append({'material':'polyimide','thick':15e-6,'n':15})
    #layers.append({'material':'silicon_nitride','thick':3e-6,'n':1})
    #layers.append({'material':'silicon','thick':725e-6,'n':29})
    layers.append({'material':'tim','thick':100e-6,'n':10})
    layers.append({'material':'copper','thick':3000e-6,'n':10})

    # bottom layer first
    layers.reverse()

    # lists of materials, thicknesses and cells
    materials = [layer['material'] for layer in layers]
    thicknesses = [layer['thick'] for layer in layers]
    ns = [layer['n'] for layer in layers]

    # y parameters are defined already, here are the x-parameters
    Lx = 1e-3
    nx = 201
    xc = 100 # the x-location of the heater

    # derived parameters
    dx = Lx/(nx-1)
    ny = sum(ns)
    dys = [layer['thick']/layer['n'] for layer in layers]
    n_bound = [sum(ns[:i]) for i in range(len(ns)+1)]

    # initialise arrays
    k_out, c_out, rho_out, q_out, v_out = np.zeros((nx,ny)), np.zeros((nx,ny)), np.zeros((nx,ny)), np.zeros((nx,ny)), np.zeros((nx,ny))
    dy = np.zeros(ny)

    # populate the material stack
    for i,material in enumerate(materials):
        c_out[:,n_bound[i]:n_bound[i+1]] = c_dict[material]
        rho_out[:, n_bound[i]:n_bound[i + 1]] = rho_dict[material]
        k_out[:, n_bound[i]:n_bound[i + 1]] = k_dict[material]
        dy[n_bound[i]:n_bound[i + 1]] = dys[i]

    # update the arrays to include the heater
    c_out, rho_out, k_out, q_out = heater_continuous(c_out, rho_out, k_out, q_out, xc, c_dict, rho_dict, k_dict,
                                          n_bound, power, vias, heater_metal,dx,dys)
    # populate the velocity array
    n = 6
    for yy in range(n_bound[n],n_bound[n+1]):
        v_out[:, yy] = 4 * v_max * (n_bound[n+1] - yy) * (n_bound[n] - yy) / (n_bound[n+1] - n_bound[n]) ** 2

    # create the dimension vectors
    x_lim = (nx - 1) * dx / 2
    x = np.linspace(-x_lim, x_lim, nx)

    # the y array gives a point in the middle of each quad in the y-direction
    dy_helper = [dy[i]/2 + dy[i+1]/2 for i in range(len(dy)-1)]
    y = np.array([dy[0]/2 + sum(dy[:i]) for i in range(len(dy))])

    return (k_out,c_out,rho_out,q_out,v_out,x,y,dx,dy,nx,ny,n_bound)


def model_3_sites(wet = True, vias = True, power = [1,1,1], heater_metal = True, v_max = 0):
    '''
    Model of the polyimide chip.

    Notes
    1. The 1 um Ti + Ag back side metallisation is not included. It will have around the same thermal parameters as
    Si and is only 1 um thick by comparision with the 725 um of the Si wafer. Conclusion. Negligable.
    2. The 4 um patterned Al layer on the top of the silicon wafer is not included. Again it has around the
    same thermal parameters as Si and is relatively thin by comparision with the Si wafer.
    3. The ruthenium layer is included, but only in the region of the heater where the areal fill factor is greatest.

    Parameters
    ----------
    wet boolean: is there water or air above the chip?
    vias boolean: are the through polyimide vias included
    power float: the power/unit length for each quad in the simulation
    heater_metal boolean: is the ruthenium heater metal included in the simulation?
    v_max float: the velocity in the centre of the flow profile, i.e. the maximum

    Returns
    ----------
    The function returns the following tuple (k_out,c_out,rho_out,q_out,v_out,x,y,dx,dy,nx,ny)


    '''
    # user defined parameters
    # the materials stack from top to bottom including materials, thicknesses and number of cells in the
    # vertical direction in each layer
    layers = []
    layers.append({'material': 'silicon_dioxide', 'thick': 1000e-6, 'n': 10})
    if wet:
        layers.append({'material': 'water', 'thick': 98e-6, 'n': 49})
        layers.append({'material': 'water', 'thick': 2e-6, 'n': 1})
    else:
        layers.append({'material': 'air', 'thick': 98e-6, 'n': 49})
        layers.append({'material': 'air', 'thick': 2e-6, 'n': 1})
    layers.append({'material':'silicon_nitride', 'thick': 7e-7, 'n': 1})
    layers.append({'material':'silicon_nitride','thick':3e-7,'n':1})
    layers.append({'material':'silicon_nitride','thick':1e-6,'n':1})
    layers.append({'material':'polyimide','thick':10e-6,'n':10})
    layers.append({'material':'polyimide','thick':15e-6,'n':15})
    layers.append({'material':'silicon_nitride','thick':3e-6,'n':1})
    layers.append({'material':'silicon','thick':725e-6,'n':29})
    layers.append({'material':'tim','thick':100e-6,'n':10})
    layers.append({'material':'copper','thick':3000e-6,'n':10})

    # bottom layer first
    layers.reverse()

    # lists of materials, thicknesses and cells
    materials = [layer['material'] for layer in layers]
    thicknesses = [layer['thick'] for layer in layers]
    ns = [layer['n'] for layer in layers]

    # y parameters are defined already, here are the x-parameters
    Lx = 2e-3
    nx = 401
    xc = 200 # the x-location of the heater

    # derived parameters
    dx = Lx/(nx-1)
    ny = sum(ns)
    dys = [layer['thick']/layer['n'] for layer in layers]
    n_bound = [sum(ns[:i]) for i in range(len(ns)+1)]

    # initialise arrays
    k_out, c_out, rho_out, q_out, v_out = np.zeros((nx,ny)), np.zeros((nx,ny)), np.zeros((nx,ny)), np.zeros((nx,ny)), np.zeros((nx,ny))
    dy = np.zeros(ny)

    # populate the material stack
    for i,material in enumerate(materials):
        c_out[:,n_bound[i]:n_bound[i+1]] = c_dict[material]
        rho_out[:, n_bound[i]:n_bound[i + 1]] = rho_dict[material]
        k_out[:, n_bound[i]:n_bound[i + 1]] = k_dict[material]
        dy[n_bound[i]:n_bound[i + 1]] = dys[i]

    # update the arrays to include the heater
    c_out, rho_out, k_out, q_out = heater_continuous(c_out, rho_out, k_out, q_out, xc, c_dict, rho_dict, k_dict,
                                          n_bound, power[1], vias, heater_metal,dx,dys)

    c_out, rho_out, k_out, q_out = heater_continuous(c_out, rho_out, k_out, q_out, xc - 80, c_dict, rho_dict, k_dict,
                                                     n_bound, power[0], vias, heater_metal,dx,dys)

    c_out, rho_out, k_out, q_out = heater_continuous(c_out, rho_out, k_out, q_out, xc + 80, c_dict, rho_dict, k_dict,
                                                     n_bound, power[2], vias, heater_metal,dx,dys)

    # This is a kludge, we assume that the heater is only in layer 7, the heater metal and need the density
    # q_out = q_out/dx/dys[7]

    # populate the velocity array
    n = 10
    for yy in range(n_bound[n],n_bound[n+1]):
        v_out[:, yy] = 4 * v_max * (n_bound[n+1] - yy) * (n_bound[n] - yy) / (n_bound[n+1] - n_bound[n]) ** 2

    # create the dimension vectors
    x_lim = (nx - 1) * dx / 2
    x = np.linspace(-x_lim, x_lim, nx)

    # the y array gives a point in the middle of each quad in the y-direction
    dy_helper = [dy[i]/2 + dy[i+1]/2 for i in range(len(dy)-1)]
    y = np.array([dy[0]/2 + sum(dy[:i]) for i in range(len(dy))])

    return (k_out,c_out,rho_out,q_out,v_out,x,y,dx,dy,nx,ny,n_bound)
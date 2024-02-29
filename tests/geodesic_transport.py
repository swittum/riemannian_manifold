import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from riemannian_manifold import RiemannianManifold

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def to_cartesian(theta, phi):
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def translate_tangent(theta, phi, vector):
    x = np.cos(theta)*np.cos(phi)*vector[0]-np.sin(theta)*np.sin(phi)*vector[1]
    y = np.cos(theta)*np.sin(phi)*vector[0]+np.sin(theta)*np.cos(phi)*vector[1]
    z = -np.sin(theta)*vector[0]
    return x, y, z


r, theta, phi = sp.symbols('r theta phi')
g = sp.diag(r**2, r**2*sp.sin(theta)**2)
coords = [theta, phi]
S2 = RiemannianManifold(coords, g)


tau = sp.symbols('tau')
V0 = [1, 0]


# Part 1
curve = sp.Matrix([tau, 0])
interval1 = np.arange(0.01, np.pi/2, 0.01)
solution_ar1 = S2.parallel_transport(V0, curve, tau, interval1, params={r: 1})
solution = solution_ar1[-1, :]


# Part 2
curve = sp.Matrix([sp.pi/2, tau])
interval2 = np.arange(0, np.pi/2, 0.01)
solution_ar2 = S2.parallel_transport(solution, curve, tau, interval2, params={r: 1})
solution = solution_ar2[-1, :]


# Part 3
curve = sp.Matrix([sp.pi/2-tau, np.pi/2])
interval3 = np.arange(0, np.pi/2-0.01, 0.01)
solution_ar3 = S2.parallel_transport(solution, curve, tau, interval3, params={r: 1})
solution = solution_ar3[-1, :]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='b', alpha=0.5)


# Visualizing Curves


# Curve 1
tau1 = interval1
theta1 = tau1
phi1 = np.zeros(len(tau1))
curve1 = np.array([theta1, phi1])
curve1_cart = to_cartesian(curve1[0], curve1[1])
ax.plot(curve1_cart[0], curve1_cart[1], curve1_cart[2], color='r')
for i in range(0, len(tau1), 10):
    position = to_cartesian(*curve1[:, i])
    tangent_polar = solution_ar1[i, :]
    tangent_cartesian = translate_tangent(*curve1[:, i], tangent_polar)
    ax.quiver(position[0], position[1], position[2], tangent_cartesian[0], tangent_cartesian[1], tangent_cartesian[2], color='g')


# Curve 2
tau2 = interval2
theta2 = np.pi/2*np.ones(len(tau2))
phi2 = tau2
curve2 = np.array([theta2, phi2])
curve2_cart = to_cartesian(curve2[0], curve2[1])
ax.plot(curve2_cart[0], curve2_cart[1], curve2_cart[2], color='r')
for i in range(0, len(tau2), 10):
    position = to_cartesian(*curve2[:, i])
    tangent_polar = solution_ar2[i, :]
    tangent_cartesian = translate_tangent(*curve2[:, i], tangent_polar)
    ax.quiver(position[0], position[1], position[2], tangent_cartesian[0], tangent_cartesian[1], tangent_cartesian[2], color='g')


# Curve 3
tau3 = interval3
theta3 = np.pi/2-tau3
phi3 = np.pi/2*np.ones(len(tau3))
curve3 = np.array([theta3, phi3])
curve3_cart = to_cartesian(curve3[0], curve3[1])
ax.plot(curve3_cart[0], curve3_cart[1], curve3_cart[2], color='r')
for i in range(0, len(tau3), 10):
    position = to_cartesian(*curve3[:, i])
    tangent_polar = solution_ar3[i, :]
    tangent_cartesian = translate_tangent(*curve3[:, i], tangent_polar)
    ax.quiver(position[0], position[1], position[2], tangent_cartesian[0], tangent_cartesian[1], tangent_cartesian[2], color='g')


plt.show()
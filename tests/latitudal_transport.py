import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from riemannian_manifold import RiemannianManifold

import numpy as np
import sympy as sp 
import matplotlib.pyplot as plt


r, theta, phi = sp.symbols('r theta phi')
g = sp.diag(r**2, r**2*sp.sin(theta)**2)
coords = [theta, phi]
S2 = RiemannianManifold(coords, g)

tau = sp.symbols('tau')
theta0 = np.pi/4
curve = sp.Matrix([theta0, tau])
interval = np.arange(0, 2*np.pi, 0.01)
V0 = [1, 0]

solution = S2.parallel_transport(V0, curve, tau, interval, params={r: 1})

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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='b', alpha=0.1)

tau = interval
n = len(tau)
theta = np.ones(n)*theta0
phi = tau
curve = np.array([theta, phi])
curve_cartesian = np.array([to_cartesian(theta[i], phi[i]) for i in range(n)])
ax.plot(curve_cartesian[:, 0], curve_cartesian[:, 1], curve_cartesian[:, 2], 'r')

for i in range(0, n-1, 100):
    print(i)
    position = to_cartesian(*curve[:, i])
    tangent_polar = solution[i, :]
    tangent_cartesian = translate_tangent(*curve[:, i], tangent_polar)
    ax.quiver(position[0], position[1], position[2], tangent_cartesian[0], tangent_cartesian[1], tangent_cartesian[2], color='g')

plt.show()
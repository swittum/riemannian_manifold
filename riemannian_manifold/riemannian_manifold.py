import sympy as sp
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize


class RiemannianManifold:

    
    def __init__(self, coords, g):
        """
        Class to model a Riemannian manifold and compute its basic features
        :param coords: list of symbols representing the coordinates of the manifold
        :param g: the metric tensor of the manifold
        """
        self.coords = coords
        self.g = g
        self.g_inv = g.inv()
        self.dimensions = len(coords)
        self.Christoffels = None
        self.RiemannTensor = None
        self.RicciTensor = None
        self.RicciScalar = None
        self.EinsteinTensor = None
        self.geodesic_eqs = None


    def compute_inner_product(self, point, v1, v2, params={}):
        """
        Compute the inner product of two vectors at a given point
        :param point: list of numbers representing the coordinates of the point
        :param v1: list of numbers representing the components of the first vector
        :param v2: list of numbers representing the components of the second vector
        :return: the inner product of v1 and v2 at the point
        """
        if type(point) is dict:
            g = self.g.subs(point)
        else:
            substitutions = {self.coords[i]: point[i] for i in range(self.dimensions)}
            g = self.g.subs(substitutions)
        g = np.array(g.subs(params)).astype(float)
        # g = np.array(g).astype(float)
        out = v1.dot(g.dot(v2))
        return out 


    def get_christoffels(self):
        """
        Compute the Christoffel symbols of the manifold
        :return: a dictionary with the Christoffel symbols as elements
        """
        self.Christoffels = {}
        for mu in range(self.dimensions):
            Gamma = sp.zeros(self.dimensions, self.dimensions)
            for alpha in range(self.dimensions):
                for beta in range(self.dimensions):
                    for delta in range(self.dimensions):
                        Gamma[alpha, beta] += sp.Rational(1, 2)*self.g_inv[mu, delta]*(sp.diff(self.g[beta, delta], self.coords[alpha])+sp.diff(self.g[delta, alpha], self.coords[beta])-sp.diff(self.g[alpha, beta], self.coords[delta]))
            self.Christoffels[self.coords[mu]] = sp.simplify(Gamma)
        return self.Christoffels


    def get_riemann_tensor(self):
        """
        Compute the Riemann tensor of the manifold
        :return: a dictionary with the Riemann tensor as elements
        """
        if self.RiemannTensor is None:
            Christoffels = self.get_christoffels()
            self.RiemannTensor = {}
            for rho in range(self.dimensions):
                self.RiemannTensor[self.coords[rho]] = {}
                for sigma in range(self.dimensions):
                    tmp = sp.zeros(self.dimensions, self.dimensions)
                    for mu in range(self.dimensions):
                        for nu in range(self.dimensions):
                            tmp[mu, nu] = sp.diff(Christoffels[self.coords[rho]][nu, sigma], self.coords[mu])-sp.diff(Christoffels[self.coords[rho]][sigma, mu], self.coords[nu])
                            for lambda_ in range(self.dimensions):
                                tmp[mu, nu] += Christoffels[self.coords[rho]][mu, lambda_]*Christoffels[self.coords[lambda_]][sigma, nu]-Christoffels[self.coords[rho]][nu, lambda_]*Christoffels[self.coords[lambda_]][mu, sigma]
                    self.RiemannTensor[self.coords[rho]][self.coords[sigma]] = sp.simplify(tmp)
        return self.RiemannTensor


    def get_ricci_tensor(self):
        """
        Compute the Ricci tensor of the manifold
        :return: the Ricci tensor of the manifold
        """
        R = self.get_riemann_tensor()
        self.RicciTensor = sp.zeros(self.dimensions, self.dimensions)
        for mu in range(self.dimensions):
            for nu in range(self.dimensions):
                for rho in range(self.dimensions):
                    self.RicciTensor[mu, nu] += R[self.coords[rho]][self.coords[mu]][rho, nu]
        self.RicciTensor = sp.simplify(self.RicciTensor)
        return self.RicciTensor


    def get_ricci_scalar(self):
        """
        Compute the Ricci scalar of the manifold
        :return: the Ricci scalar of the manifold
        """
        Ricci = self.get_ricci_tensor()
        g_inv = self.g.inv()
        tmp = g_inv*Ricci
        self.RicciScalar = 0
        for i in range(self.dimensions):
            self.RicciScalar += tmp[i, i]
        self.RicciScalar = sp.simplify(self.RicciScalar)
        return self.RicciScalar
    

    def get_einstein_tensor(self):
        """
        Compute the Einstein tensor of the manifold
        :return: the Einstein tensor of the manifold
        """
        RicciTensor = self.get_ricci_tensor()
        RicciScalar = self.get_ricci_scalar()
        self.EinsteinTensor = RicciTensor-sp.Rational(1, 2)*self.g*RicciScalar
        self.EinsteinTensor = sp.simplify(self.EinsteinTensor)
        return self.EinsteinTensor


    def _set_up_curve(self, param):
        """
        Set up a curve in the manifold
        :param param: the parameter of the curve
        :return: a list of symbols representing the components of the curve and a dictionary with the substitutions
        """
        t = sp.symbols(param)
        gamma = []
        for i in range(self.dimensions):
            gamma.append(sp.Function(f'gamma_{i}')(t))
        substitutions = {self.coords[i]: gamma[i] for i in range(self.dimensions)}
        return gamma, substitutions


    def get_geodesic_eqs(self):
        """
        Compute the geodesic equations of the manifold
        :return: the geodesic equations of the manifold
        """
        gamma, substitutions = self._set_up_curve('tau')
        if not self.Christoffels:
            self.get_christoffels()
        Christoffels_subs = {self.coords[i]: self.Christoffels[self.coords[i]].subs(substitutions) for i in range(self.dimensions)}
        geodesic_eqs = {}
        for i in range(self.dimensions):
            eq = sp.diff(gamma[i], 'tau', 2)
            for j in range(self.dimensions):
                for k in range(self.dimensions):
                    eq += Christoffels_subs[self.coords[i]][j, k]*sp.diff(gamma[j], 'tau')*sp.diff(gamma[k], 'tau')
            eq = sp.simplify(eq)
            geodesic_eqs[self.coords[i]] = eq
        self.geodesic_eqs = geodesic_eqs
        return self.geodesic_eqs
    

    def get_geodesic(self, start, end, step=0.01, params={}):
        """
        Compute the geodesic between two points numerically
        :param start: list of numbers representing the coordinates of the starting point
        :param end: list of numbers representing the coordinates of the ending point
        :param step: the step size for the numerical integration
        :param params: dictionary with the values of the parameters of the manifold
        """
        if not self.geodesic_eqs:
            self.get_geodesic_eqs()
        geodesic_eqs = self.geodesic_eqs
        gamma, _ = self._set_up_curve('tau')
        dd_gamma = [gamma[i].diff('tau', 2) for i in range(self.dimensions)]
        geodesic_eqs_resolved = {}
        for i in range(self.dimensions):
            eq = sp.Eq(geodesic_eqs[self.coords[i]], 0)
            sol = sp.solve(eq, dd_gamma[i])[0]
            geodesic_eqs_resolved[self.coords[i]] = sol
        d_coords = [sp.symbols(coord.name+"^'") for coord in self.coords]
        substitutions1 = {gamma[i].diff('tau'): d_coords[i] for i in range(self.dimensions)}
        substitutions2 = {gamma[i]: self.coords[i] for i in range(self.dimensions)}
        funcs = []
        arguments = [*self.coords, *d_coords, 'tau']
        for i in range(self.dimensions):
            geodesic_eqs_resolved[self.coords[i]] = geodesic_eqs_resolved[self.coords[i]].subs(params).subs(substitutions1)
            geodesic_eqs_resolved[self.coords[i]] = geodesic_eqs_resolved[self.coords[i]].subs(params).subs(substitutions2)
            funcs.append(sp.lambdify(arguments, d_coords[i], 'numpy'))
        for i in range(self.dimensions):
            funcs.append(sp.lambdify(arguments, geodesic_eqs_resolved[self.coords[i]], 'numpy')) 
        def dy_dt(y, t):
            out = [func(*y, t) for func in funcs]
            return out
        t = np.arange(0, 1, step)
        def loss(v):
            sol = odeint(dy_dt, [*start, *v], t)[:, :self.dimensions]
            loss = 0
            for i in range(self.dimensions):
                loss += (sol[-1, i]-end[i])**2
            return loss
        init = [1 for _ in range(self.dimensions)]
        result = minimize(loss, init)
        # print("The minimization terminated with a loss of ", result.fun)
        solution = odeint(dy_dt, [*start, *result.x], t)
        positions = solution[:, :self.dimensions]
        velocities = solution[:, self.dimensions:]
        # Reparameterize by proper time
        norm = np.sqrt(np.abs(self.compute_inner_product(
            np.array(start), np.array(result.x), np.array(result.x), params=params)))
        v_normalized = [np.array(result.x)[i]/norm for i in range(self.dimensions)]
        t = np.arange(0, norm, step)
        solution = odeint(dy_dt, [*start, *v_normalized], t)
        positions_normalized = solution[:, :self.dimensions]
        # velocities_normalized = solution[:, self.dimensions:]
        return positions_normalized, norm
    

    def parallel_transport(self, V0, curve, tau, interval, params={}):
        """
        Compute the parallel transport of a vector along a curve
        :param V0: the initial vector
        :param curve: the curve along which the vector is parallel transported
        :param tau: the parameter of the curve
        :param interval: the interval of the parameter
        :param params: dictionary with the values of the parameters of the manifold
        :return: the parallel transported vector
        """
        if not self.Christoffels:
            self.get_christoffels()
        substitutions = {self.coords[i]: curve[i] for i in range(self.dimensions)}
        Christoffels_subs = {self.coords[i]: self.Christoffels[self.coords[i]].subs(substitutions).subs(params) for i in range(self.dimensions)}
        V = sp.Matrix([sp.symbols(f'V_{i}') for i in range(self.dimensions)])
        d_curve = curve.diff()
        rhs = []
        for mu in range(self.dimensions):
            eq = 0
            for alpha in range(self.dimensions):
                for beta in range(self.dimensions):
                    eq += -d_curve[alpha]*Christoffels_subs[self.coords[mu]][alpha, beta]*V[beta]
            rhs.append(eq)
        dy_dt = sp.lambdify([V, tau], [rhs_.rewrite(sp.tan) for rhs_ in rhs], 'numpy')
        solution = odeint(dy_dt, V0, interval[:-1])
        return solution
    

    def covariant_derivative(self, T):
        # TODO: Implement the covariant derivative of a tensor. In order to do so, we need to implement
        # the notion of tensor fields first.
        pass
    

class ChartTransition:


    def __init__(self, old_coords, new_coords):
        self.old_coords = old_coords
        self.new_coords = new_coords
        self.dimensions = len(old_coords)
        self.J = None
        self.partials = None


    def get_jacobian(self):
        J = sp.zeros(self.dimensions, self.dimensions)
        for i in range(self.dimensions):
            for j in range(self.dimensions):
                J[i, j] = sp.diff(self.old_coords[i], self.new_coords[j])
        self.J = J
        return J
    

    def evaluate_jacobian(self, point, params={}):
        if not self.J:
            self.get_jacobian()
        J = self.J.subs({self.new_coords[i]: point[i] for i in range(self.dimensions)}).subs(params)
        return np.array(J).astype(float)
    

    def get_partials(self):
        if not self.J:
            self.get_jacobian()
        J = self.J
        self.partials = {self.new_coords[i]: J[:, i] for i in range(self.dimensions)}

    
    def evaluate_partials(self, point, params={}):
        if not self.partials:
            self.get_partials()
        substitutions = {self.new_coords[i]: point[i] for i in range(self.dimensions)}
        partials = [self.partials[i].subs(substitutions).subs(params) for i in self.new_coords]
        return partials
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    r, theta, phi = sp.symbols('r theta phi')
    g = sp.diag(r**2, r**2*sp.sin(theta)**2)
    coords = [theta, phi]
    S2 = RiemannianManifold(coords, g)

    interval = np.arange(0, 2*np.pi, 0.01)
    theta0 = np.pi/4
    gamma = lambda t: np.array([theta0, t])
    V0 = [1, 0]

    old_coords = sp.Matrix([r*sp.sin(theta)*sp.cos(phi), r*sp.sin(theta)*sp.sin(phi), r*sp.cos(theta)])
    new_coords = sp.Matrix([r, theta, phi])
    chart = ChartTransition(old_coords, new_coords)
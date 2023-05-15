import numpy as np
import scipy.linalg as la
import pickle


class LinearSystemSolver:
    def __init__(self):
        """
        Initialize the constraint matrix A and matrices Q, R for the QR decomposition of A.
        """
        self.A = None
        self.Q = None
        self.R = None

    def initialize(self, A):
        """
        Initialize the solver with a fixed constraint matrix A and decompose it.
        """
        self.A = A
        self.Q, self.R = la.qr(self.A, mode='economic')
    def save_state(self, filename):
        """
        Save the state of the initialized linear system to a file.
        """
        with open(filename, 'wb') as file:
            pickle.dump((self.A, self.Q, self.R), file)

    def load_state(self, filename):
        """
        Load the saved state of a linear system from a file.
        """
        with open(filename, 'rb') as file:
            self.A, self.Q, self.R = pickle.load(file)


    def solve(self, b):
        """
        Accept the right-hand side b and compute the least-squares optimal x for the given b.
        """
        x = la.lstsq(self.R, np.dot(self.Q.T, b), lapack_driver='gelsy')[0]
        return x


    def compute_residual_norm(self, x, b):
        """
        Compute the norm of the residual ||Ax − b||^2.
        """
        Ax = np.dot(self.A, x)
        residual = Ax - b
        norm_squared = np.sum(residual ** 2)
        return norm_squared

    def check_solution(self, x, b, rtol=1e-6, atol=1e-6):
        """
        Check if the computed solution satisfies the given linear system within the specified tolerances.
        Handle the effect of floating point accuracy
        """
        Ax = np.dot(self.A, x)
        residual = Ax - b
        norm_squared = np.sum(residual ** 2)

        return np.isclose(norm_squared, 0, rtol=rtol, atol=atol)


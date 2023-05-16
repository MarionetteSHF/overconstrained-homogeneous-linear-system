import numpy as np
import scipy.linalg as la
import pickle


class LinearSystemSolver:
    def __init__(self):
        """
        Initialize the dictionaries for storing matrices Q, R from the QR decomposition of A.
        """

        self.Q_matrices = {}
        self.R_matrices = {}

    def initialize(self, A_list):
        """
        Initialize the solver by decomposing every matrix and store the Q R matrix.
        """
        for matrix_A in A_list:
            matrix_A_str = np.array2string(matrix_A)
            Q, R = la.qr(matrix_A, mode='economic')
            self.Q_matrices[matrix_A_str] = Q
            self.R_matrices[matrix_A_str] = R
    def save_state(self, filename):
        """
        Save the state of the initialized linear system to a file.
        """
        with open(filename, 'wb') as file:
            pickle.dump((self.Q_matrices, self.R_matrices), file)

    def load_state(self, filename):
        """
        Load the saved state of a linear system from a file.
        """
        with open(filename, 'rb') as file:
            self.Q_matrices, self.R_matrices = pickle.load(file)


    def solve(self, A, b):
        """
        Accept the right-hand side b and compute the least-squares optimal x for the given A and b.
        """
        matrix_A_str = np.array2string(A)
        Q = self.Q_matrices[matrix_A_str]
        R = self.R_matrices[matrix_A_str]
        x = la.lstsq(R, np.dot(Q.T, b), lapack_driver='gelsy')[0]
        return x


    def compute_residual_norm(self, A, x, b):
        """
        Compute the norm of the residual ||Ax − b||^2 for the given A, x, and b.
        """
        Ax = np.dot(A, x)
        residual = Ax - b
        norm_squared = np.sum(residual ** 2)
        return norm_squared

    def check_solution(self, A, x, b, rtol=1e-6, atol=1e-6):
        """
        Check if the computed solution satisfies the given linear system within the specified tolerances.
        Handle the effect of floating point accuracy.
        """
        Ax = np.dot(A, x)
        residual = Ax - b
        norm_squared = np.sum(residual ** 2)

        return np.isclose(norm_squared, 0, rtol=rtol, atol=atol)


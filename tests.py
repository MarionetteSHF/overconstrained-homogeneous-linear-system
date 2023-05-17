import oh_linear_system_module as oh
import numpy as np

def run_test_case(A_list, b):
    solver = oh.LinearSystemSolver()
    solver.initialize(A_list)
    for A in A_list:
        x = solver.solve(A, b)
        residual_norm = solver.compute_residual_norm(A, x, b)

        print("Constraint matrix A:")
        print(A)
        print("Right-hand side vector b:")
        print(b)
        print("Solution x:")
        print(x)
        print("Residual norm:", residual_norm)

        # Test save_state and load_state
        solver.save_state("linear_system_state.pkl")
        loaded_solver = oh.LinearSystemSolver()
        loaded_solver.load_state("linear_system_state.pkl")
        loaded_x = loaded_solver.solve(A,b)
        loaded_residual_norm = loaded_solver.compute_residual_norm(A, loaded_x, b)
        print("Loaded solution x:")
        print(loaded_x)
        print("Loaded residual norm:", loaded_residual_norm)

        # Compare original and loaded solution and residual norm
        np.testing.assert_allclose(x, loaded_x, rtol=1e-6, atol=1e-6)
        assert np.isclose(residual_norm, loaded_residual_norm, rtol=1e-6, atol=1e-6)
        print("Original and loaded solutions and residual norms match.")

# Test Case 1
A1 = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]])
b1 = np.zeros(4)
run_test_case(A1, b1)

# Test Case 2
A2 = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]]])
b2 = np.array([4, 3, 7, 5])
run_test_case(A2, b2)

# Test Case 3
A3 = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]]])
b3 = np.array([1, 1, 1, 1])
run_test_case(A3, b3)

# Test Case 4
A4 = np.array([[[2, 4, 6], [1, 2, 3], [7, 8, 9],[1, 2, 3]]])
b4 = np.array([3, 5, 7, 2])
run_test_case(A4, b4)

# Test Case 5
A5 = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
b5 = np.array([1, 2, 3])
run_test_case(A5, b5)

# Test Case 6. Multiple matrix A
A6 = np.array([[[1, 2], [3, 4], [5, 6]],[[1, 2], [1, 2], [1, 2]]])
b6 =  np.array([4, 5, 6])
run_test_case(A6, b6)



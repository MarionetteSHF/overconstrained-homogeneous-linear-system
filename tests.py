import oh_linear_system_module as oh
import numpy as np

def run_test_case(A, b):
    solver = oh.LinearSystemSolver()
    solver.initialize(A)
    x = solver.solve(b)
    residual_norm = solver.compute_residual_norm(x, b)

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
    loaded_x = loaded_solver.solve(b)
    loaded_residual_norm = loaded_solver.compute_residual_norm(loaded_x, b)
    print("Loaded solution x:")
    print(loaded_x)
    print("Loaded residual norm:", loaded_residual_norm)

    # Compare original and loaded solution and residual norm
    np.testing.assert_allclose(x, loaded_x, rtol=1e-6, atol=1e-6)
    assert np.isclose(residual_norm, loaded_residual_norm, rtol=1e-6, atol=1e-6)
    print("Original and loaded solutions and residual norms match.")

# Test Case 1
A1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
b1 = np.zeros(4)
run_test_case(A1, b1)

# Test Case 2
A2 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
b2 = np.array([4, 3, 7, 5])
run_test_case(A2, b2)

# Test Case 3
A3 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
b3 = np.array([1, 1, 1, 1])
run_test_case(A3, b3)

# Test Case 4
A4 = np.array([[2, 4, 6], [1, 2, 3], [7, 8, 9],[1, 2, 3]])
b4 = np.array([3, 5, 7, 2])
run_test_case(A4, b4)

# Test Case 5
A5 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b5 = np.array([1, 2, 3])
run_test_case(A5, b5)

# Test Case 6
A6 = np.array([[1, 0], [0, 1]])
b6 = np.array([2, 3])
run_test_case(A6, b6)

# Test Case 7
A7 = np.array([[2, 4], [6, 8]])
b7 = np.array([5, 10])
run_test_case(A7, b7)

# Test Case 8
A8 = np.array([[1, 1], [0, 0]])
b8 = np.array([2, 0])
run_test_case(A8, b8)

# Test Case 9
A9 = np.array([[1, 2, 3], [4, 5, 6]])
b9 = np.array([1, 1])
run_test_case(A9, b9)

# Test Case 10
A10 = np.array([[1, 2], [1, 2], [1, 2]])
b10 = np.array([1, 2, 3])
run_test_case(A10, b10)

# Test Case 11. Multiple right-hand sides b
A = np.array([[1, 2], [3, 4], [5, 6]])
b_list = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]

solver = oh.LinearSystemSolver()
solver.initialize(A)

solutions = []

for b in b_list:
    x = solver.solve(b)
    solutions.append(x)
    print(solver.check_solution(x, b))
# Print solutions
for i, x in enumerate(solutions):
    print(f"Solution for b{i+1}:")
    print(x)


# math_utils.py


import numpy as np
import casadi as ca


def getMRow(M):
    if isinstance(M, np.ndarray):
        return M.shape[0]
    elif isinstance(M, (ca.SX, ca.MX)):
        return M.size1()
    else:
        raise TypeError("Unsupported matrix type: expected NumPy or CasADi SX/MX.")


def getMColumn(M):
    if isinstance(M, np.ndarray):
        return M.shape[1]
    elif isinstance(M, (ca.SX, ca.MX)):
        return M.size2()
    else:
        raise TypeError("Unsupported matrix type: expected NumPy or CasADi SX/MX.")



# # test for np matrix
# A = np.zeros((3, 4))

# # test 1
# print("Rows: ", getMRow(A)) # -> expected output: "Rows: 3"

# # test 2
# print("Columns: ", getMColumn(A)) # -> expected output: "Columns: 4"


# # test for sx matrix
# B = ca.SX.zeros(3, 4)

# # test 1
# print("Rows: ", getMRow(B)) # -> expected output: "Rows: 3"

# # test 2
# print("Columns: ", getMColumn(B)) # -> expected output: "Columns: 4"


# test for sx matrix
# C = ca.MX.zeros(3, 4)

# # test 1
# print("Rows: ", getMRow(C)) # -> expected output: "Rows: 3"

# # test 2
# print("Columns: ", getMColumn(C)) # -> expected output: "Columns: 4"
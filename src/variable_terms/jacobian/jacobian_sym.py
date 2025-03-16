import sympy as sy

def func(x, y):
    return sy.Matrix([x**2 + y**2, 2*x + 3*y + 5])

x, y = sy.symbols('x y', real=True)
f = func(x, y)

# Jacobian
J = sy.Matrix([
    sy.diff(f[0], x), sy.diff(f[0], y),
    sy.diff(f[1], x), sy.diff(f[1], y)
]).reshape(2,2)

z = sy.Matrix([x, y])
J = f.jacobian(z)

print(J)

# (1,2)
# J_sym = J.subs([x, 1], [y, 2])
# or
J_sym = J.subs([(x, 1), (y, 2)])
print(J_sym)
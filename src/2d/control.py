# control.py

from kinematics import forward_kinematics


def add_friction_cone_constraint(opti, U, k, mu, leg):
    """
    Adds a Coulomb friction cone constraint for the specified leg at timestep k.

    Parameters
    ----------
    opti : casadi.Opti
        The CasADi optimization object.
    U : casadi.MX or SX
        The control input matrix of shape (nu, N-1).
    k : int
        The timestep index.
    mu : float
        Coefficient of friction.
    leg : str
        Either "front" or "rear", indicating which foot to apply the constraint to.
    """
    leg_indices = {
        "front": (0, 1),  # (Fx, Fy) for front foot
        "rear":  (2, 3),  # (Fx, Fy) for rear foot
    }

    if leg not in leg_indices:
        raise ValueError(f"Invalid leg name '{leg}'. Must be 'front' or 'rear'.")

    fx_idx, fy_idx = leg_indices[leg]
    fx = U[fx_idx, k]
    fy = U[fy_idx, k]
    
    # Friction cone inequality contraints: |Fx| ≤ μ Fy
    opti.subject_to(fx >= -mu * fy)
    opti.subject_to(fx <= mu * fy)


def add_zero_force_constraint(opti, U, k, leg):
    leg_indices = {
        "front": (0, 1),  # (Fx, Fy) for front foot
        "rear":  (2, 3),  # (Fx, Fy) for rear foot
    }

    if leg not in leg_indices:
        raise ValueError(f"Invalid leg name '{leg}'. Must be 'front' or 'rear'.")

    fx_idx, fy_idx = leg_indices[leg]
    fx = U[fx_idx, k]
    fy = U[fy_idx, k]

    opti.subject_to(fx == 0)
    opti.subject_to(fy == 0)


def add_contact_constraint(opti, Q, k, q_ref, leg):
    """
    Adds a foot contact position constraint for the specified leg at a given timestep.

    This constraint ensures that the specified foot (front or rear) remains fixed
    at a known position during contact — i.e., the ankle position at timestep `k`
    must match the position from a reference configuration `q_ref`. This is used 
    during stance phases to prevent the foot from sliding or drifting.

    Parameters
    ----------
    opti : casadi.Opti
        The CasADi optimization problem object.

    Q : casadi.MX or SX
        Symbolic matrix of generalized coordinates over time (shape: [nq, N]).

    k : int
        Timestep index at which to apply the constraint.

    q_ref : np.ndarray or casadi.MX
        Reference robot configuration (e.g., initial or final pose) to lock the ankle position to.

    leg : str
        The leg to apply the constraint to: either "front" or "rear".

    Raises
    ------
    ValueError
        If an invalid leg name is provided.

    Notes
    -----
    - Uses `forward_kinematics` to extract ankle position from both the reference pose (`q_ref`)
      and the symbolic trajectory at timestep `k` (`Q[:, k]`).
    - This constraint should be applied only during contact phases.
    - The constraint is hard equality: x/y ankle position must match exactly.
    """
    if leg not in ("front", "rear"):
        raise ValueError(f"Invalid leg name '{leg}'. Use 'front' or 'rear'.")

    # Get reference and symbolic foot positions
    fk_ref = forward_kinematics(q_ref)
    fk_opt = forward_kinematics(Q[:, k])

    # Dynamically access attributes based on leg name
    x_ref = getattr(fk_ref, f"{leg}_ankle_x")
    y_ref = getattr(fk_ref, f"{leg}_ankle_y")
    x_k   = getattr(fk_opt, f"{leg}_ankle_x")
    y_k   = getattr(fk_opt, f"{leg}_ankle_y")

    # Apply foot position constraint (no slipping during contact)
    opti.subject_to(x_k == x_ref)
    opti.subject_to(y_k == y_ref)


def add_dynamics_constraint(opti, Q, V, U, k, get_next_state_func):
    """
    Adds a dynamics constraint to the optimizer at timestep k using Euler integration.

    Parameters
    ----------
    opti : casadi.Opti
        The optimization problem object.
    Q : casadi.MX or SX
        Matrix of symbolic generalized positions over time (shape: [nq, N]).
    V : casadi.MX or SX
        Matrix of symbolic velocities over time (shape: [nv, N]).
    U : casadi.MX or SX
        Matrix of control inputs over time (shape: [nu, N-1]).
    k : int
        The timestep index at which to apply the constraint.
    get_next_state_func : casadi.Function
        A CasADi function that returns (q_next, v_next) from (q, v, u).
    """
    q_next, v_next = get_next_state_func(Q[:, k], V[:, k], U[:, k])
    opti.subject_to(Q[:, k + 1] == q_next)
    opti.subject_to(V[:, k + 1] == v_next)


def apply_joint_limits(opti, q_angles_diff, lower_bounds, upper_bounds):
    for i in range(q_angles_diff.shape[0]):
        opti.subject_to(q_angles_diff[i] >= lower_bounds[i])
        opti.subject_to(q_angles_diff[i] <= upper_bounds[i])


def initial_guess(opti, Q, V, U, q_initial, q_final, v_initial, v_final, m, g, N, contact):
    for i in range(N):
        # to interpolate Q and V
        omega = i / N
        q_guess = q_initial + omega * (q_final - q_initial)
        v_guess = v_initial + omega * (v_final - v_initial)

        opti.set_initial(Q[:, i], q_guess)
        opti.set_initial(V[:, i], v_guess)

    for i in range(N - 1):
        if contact(i):
            Fy_guess = m * g / 2
            opti.set_initial(U[1, i], Fy_guess) # GRF1_y
            opti.set_initial(U[3, i], Fy_guess) # GRF2_y
        else:
            opti.set_initial(U[1, i], 0) # GRF1_y
            opti.set_initial(U[3, i], 0) # GRF2_y
        
        opti.set_initial(U[0, i], Fy_guess) # GRF1_x
        opti.set_initial(U[2, i], Fy_guess) # GRF2_x

        opti.set_initial(U[4, i], 0) # dtheta1
        opti.set_initial(U[5, i], 0) # dtheta2
        opti.set_initial(U[6, i], 0) # dtheta3
        opti.set_initial(U[7, i], 0) # dtheta4
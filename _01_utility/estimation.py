import numpy as np
import cvxpy as cp

def estimate_clock_rates(network, matrices):
    n = len(network)
    K = len(matrices)
    
    # pre-built numpy matrix for speed
    rows = []
    for k in range(K - 1):
        Mk, Mkp1 = matrices[k], matrices[k+1]
        for i in range(n):
            for j in range(n):
                if i == j: continue
                d_rec = Mkp1[i, j] - Mk[i, j]
                
                row = np.zeros(n)
                row[i] = d_rec
                row[j] = -(Mkp1[i, i] - Mk[i, i])
                rows.append(row)

    A_mat = np.array(rows)
    rho = cp.Variable(n)
    
    # obj: minimize residuals
    # constr: force rho[0] to be 1
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A_mat @ rho)), 
                     [rho[0] == 1.0])
    
    prob.solve() 
    return rho.value

def estimate_clock_offsets(network, matrices, rho_hat):
    n = len(network)
    K = len(matrices)

    theta = cp.Variable(n)
    n_distances = n * (n - 1) // 2
    d_vec = cp.Variable(n_distances)

    # map (i, j) pairs to indices in d_vec
    pair_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pair_to_idx[tuple(sorted((i, j)))] = idx
            idx += 1

    A_rows = []
    h_vals = []

    for k in range(K):
        T = matrices[k]
        for i in range(n):
            for j in range(n):
                if i == j: continue
                
                rho_i, rho_j = rho_hat[i], rho_hat[j]
                rho_ji = rho_j / rho_i
                
                # h_ij_k = T_observed_at_j - (rho_j/rho_i)*T_sender_local
                h_ij_k = T[i, j] - rho_ji * T[i, i]
                h_vals.append(h_ij_k)
                
                # rho_j * d_ij + theta_j - rho_ji * theta_i = h_ij_k
                row_theta = np.zeros(n)
                row_theta[j] = 1
                row_theta[i] = -rho_ji
                
                row_d = np.zeros(n_distances)
                row_d[pair_to_idx[tuple(sorted((i, j)))]] = rho_j
                
                A_rows.append(cp.hstack([row_theta, row_d]))

    A_mat = cp.vstack(A_rows)
    h_vec = np.array(h_vals)
    
    objective = cp.Minimize(cp.sum_squares(A_mat @ cp.hstack([theta, d_vec]) - h_vec))
    
    constraints = [theta[0] == 0, d_vec >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver = 'SCS', verbose = True) # ECOS ?

    # reconst. symmetric d matrix for print loop
    d_matrix = np.zeros((n, n))
    for (i, j), idx in pair_to_idx.items():
        d_matrix[i, j] = d_matrix[j, i] = d_vec.value[idx]

    return theta.value, d_matrix

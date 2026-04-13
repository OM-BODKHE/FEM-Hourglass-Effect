import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# --- 1. Define Beam Properties ---
L = 18.0         # Length
H = 0.2          # Height
t = 0.1          # Thickness
E = 210e7        # Young's Modulus
nu = 0.3         # Poisson's Ratio
P = -1000.0      # Point load at tip

# Mesh Properties
n_el_x = 150
n_el_y = 11

# Derived Properties
I = t * H**3 / 12

# --- 2. Generate Mesh ---
n_nodes_x = n_el_x + 1
n_nodes_y = n_el_y + 1
num_nodes = n_nodes_x * n_nodes_y
num_elements = n_el_x * n_el_y

# Node coordinates [x_coord, y_coord]
nodes = np.zeros((num_nodes, 2))
node_count = 0
for i in range(n_nodes_y):
    for j in range(n_nodes_x):
        nodes[node_count, :] = [j * L / n_el_x, i * H / n_el_y - H / 2]
        node_count += 1

# Element connectivity (0-indexed) [n1, n2, n3, n4]
elements = np.zeros((num_elements, 4), dtype=int)
elem_count = 0
for i in range(n_el_y):
    for j in range(n_el_x):
        n1 = i * n_nodes_x + j
        n2 = i * n_nodes_x + j + 1
        n3 = (i + 1) * n_nodes_x + j + 1
        n4 = (i + 1) * n_nodes_x + j
        elements[elem_count, :] = [n1, n2, n3, n4]
        elem_count += 1

# --- 3. Material Constitutive Matrix (Plane Stress) ---
D = (E / (1 - nu**2)) * np.array([
    [1, nu, 0],
    [nu, 1, 0],
    [0, 0, (1 - nu) / 2]
])

# --- 4. Perform Analysis ---
integration_schemes = ['full', 'reduced']
tip_deflections = {}
displacements_map = {}

for scheme in integration_schemes:
    num_dofs = 2 * num_nodes
    K_global = np.zeros((num_dofs, num_dofs))
    F_global = np.zeros(num_dofs)
    
    # Gauss Quadrature Points and Weights
    if scheme == 'full':
        gp_val = 1/np.sqrt(3)
        gauss_points = np.array([[-gp_val, -gp_val], [gp_val, -gp_val], 
                                 [gp_val, gp_val], [-gp_val, gp_val]])
        gauss_weights = np.array([1, 1, 1, 1])
    else: # reduced
        gauss_points = np.array([[0, 0]])
        gauss_weights = np.array([4])

    # Assemble Global Stiffness Matrix
    for i in range(num_elements):
        node_ids = elements[i, :]
        elem_coords = nodes[node_ids, :]
        K_elem = np.zeros((8, 8))
        
        for gp in range(len(gauss_weights)):
            xi, eta = gauss_points[gp]
            w = gauss_weights[gp]
            
            # Shape function derivatives in natural coordinates
            dN_dxi = 0.25 * np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
            dN_deta = 0.25 * np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
            
            # Jacobian matrix
            J = np.array([
                [np.dot(dN_dxi, elem_coords[:, 0]), np.dot(dN_dxi, elem_coords[:, 1])],
                [np.dot(dN_deta, elem_coords[:, 0]), np.dot(dN_deta, elem_coords[:, 1])]
            ])
            
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)
            
            # Derivatives in physical coordinates
            dN_dxy = invJ @ np.array([dN_dxi, dN_deta])
            
            # Strain-displacement matrix (B)
            B = np.zeros((3, 8))
            B[0, 0::2] = dN_dxy[0, :]
            B[1, 1::2] = dN_dxy[1, :]
            B[2, 0::2] = dN_dxy[1, :]
            B[2, 1::2] = dN_dxy[0, :]
            
            K_elem += (B.T @ D @ B) * detJ * w * t

        # Assemble into K_global
        dof_indices = []
        for node_id in node_ids:
            dof_indices.extend([2*node_id, 2*node_id + 1])
        
        # Use meshgrid-style indexing for NumPy assembly
        ix = np.ix_(dof_indices, dof_indices)
        K_global[ix] += K_elem

    # --- Boundary Conditions and Loads ---
    # Fixed support at x = 0
    fixed_nodes = np.where(nodes[:, 0] == 0)[0]
    fixed_dofs = np.concatenate([2*fixed_nodes, 2*fixed_nodes + 1])
    
    # Point load at tip (x = L)
    tip_nodes = np.where(nodes[:, 0] == L)[0]
    F_global[2*tip_nodes + 1] = P / len(tip_nodes)
    
    # Solve
    all_dofs = np.arange(num_dofs)
    free_dofs = np.delete(all_dofs, fixed_dofs)
    
    U = np.zeros(num_dofs)
    U[free_dofs] = np.linalg.solve(K_global[np.ix_(free_dofs, free_dofs)], F_global[free_dofs])
    
    # Store Results
    center_tip_node = tip_nodes[len(tip_nodes)//2]
    tip_deflections[scheme] = U[2*center_tip_node + 1]
    displacements_map[scheme] = U

# --- 5. Analytical Solution ---
deflection_analytical = (P * L**3) / (3 * E * I)

# --- 6. Display Results ---
print(f"--- Cantilever Beam Deflection Analysis ---")
print(f"Analytical Tip Deflection: {deflection_analytical:.5f}")
for scheme in integration_schemes:
    val = tip_deflections[scheme]
    error = 100 * abs((val - deflection_analytical) / deflection_analytical)
    print(f"{scheme.capitalize()} Integration Tip Deflection: {val:.5f} (Error: {error:.2f}%)")

# --- 7. Plotting ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
scale_factor = 0.5 * abs(L / deflection_analytical)

for idx, scheme in enumerate(integration_schemes):
    ax = axes[idx]
    U = displacements_map[scheme]
    # Reshape U to add to coordinates
    deformed_nodes = nodes + U.reshape(-1, 2) * scale_factor
    
    patches_def = []
    patches_orig = []
    
    for elem in elements:
        patches_def.append(Polygon(deformed_nodes[elem], closed=True))
        patches_orig.append(Polygon(nodes[elem], closed=True))
    
    # Plot original mesh (dashed)
    p_orig = PatchCollection(patches_orig, facecolors='none', edgecolors='gray', linestyles='--', alpha=0.3)
    ax.add_collection(p_orig)
    
    # Plot deformed mesh
    color = 'blue' if scheme == 'full' else 'red'
    p_def = PatchCollection(patches_def, facecolors=color, edgecolors='black', alpha=0.5)
    ax.add_collection(p_def)
    
    ax.set_title(f"{scheme.capitalize()} Integration - Tip Deflection: {tip_deflections[scheme]:.4f}")
    ax.set_xlim(-1, L + 2)
    ax.set_ylim(-H*5, H*5)
    ax.set_aspect('equal')
    ax.grid(True)

plt.tight_layout()
plt.show()
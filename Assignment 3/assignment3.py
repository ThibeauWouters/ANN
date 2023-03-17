import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12,5)
plt.rcParams['figure.dpi'] = 300
import os
master_dir = os.getcwd()


def generate_data(dim, nb_points):
    return np.random.normal(size=(dim, nb_points))


def pca(x, threshold):

    # Zero mean the data: compute the mean
    mean = np.mean(x, axis=1)
    # x = x - mean
    # Subtract the mean
    for i in range(np.shape(x)[1]):
        x[:, i] = x[:, i] - mean

    cov = np.cov(x)
    eig_vals, eig_vectors = np.linalg.eig(cov)
    # Sort eig vals and eig vectors for convencience of plotting
    sort_indices = np.argsort(-eig_vals)
    eig_vals_sorted = eig_vals[sort_indices]
    eig_vectors_sorted = eig_vectors[sort_indices]

    cumsum = np.cumsum(eig_vals_sorted)
    filter_result = np.where(cumsum < threshold * cumsum[-1], 1, 0)
    q = np.sum(filter_result)
    print(f"The reduced dimension is {q}")
    chosen_indices = sort_indices[:q]
    chosen_vals, chosen_vectors = eig_vals[chosen_indices], eig_vectors[chosen_indices]
    # Projection matrix
    projection_matrix = chosen_vectors
    # USED????
    E = np.transpose(chosen_vectors)
    # We choose F = E here, as written in the assignment
    F = E

    z = np.matmul(projection_matrix, x)
    reconstructed = np.matmul(F, z)
    x_hat = reconstructed
    # Add the mean again:
    for i in range(np.shape(x)[1]):
        x[:, i] = x[:, i] - mean

    return x_hat, q

def pca_given_q(x, q):

    # Zero mean the data: compute the mean
    mean = np.mean(x, axis=1)
    # x = x - mean
    # Subtract the mean
    for i in range(np.shape(x)[1]):
        x[:, i] = x[:, i] - mean

    cov = np.cov(x)
    eig_vals, eig_vectors = np.linalg.eig(cov)
    # Sort eig vals and eig vectors for convencience of plotting
    sort_indices = np.argsort(-eig_vals)
    eig_vals_sorted = eig_vals[sort_indices]
    eig_vectors_sorted = eig_vectors[sort_indices]

    chosen_indices = sort_indices[:q]
    chosen_vals, chosen_vectors = eig_vals[chosen_indices], eig_vectors[chosen_indices]
    # Projection matrix
    projection_matrix = chosen_vectors
    # USED????
    E = np.transpose(chosen_vectors)
    # We choose F = E here, as written in the assignment
    F = E

    z = np.matmul(projection_matrix, x)
    reconstructed = np.matmul(F, z)
    x_hat = reconstructed
    # Add the mean again:
    for i in range(np.shape(x)[1]):
        x[:, i] = x[:, i] - mean

    return x_hat


def compute_rmsd(x, x_hat):
    return np.sqrt(np.mean((x - x_hat) ** 2))


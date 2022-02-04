import numpy as np
import itertools


# compute exact distribution from weight matrices
def compute_probabilities(num_nodes, num_classes, A):

    num_states = num_classes ** num_nodes
    probs = np.zeros(num_states)

    dim0 = num_nodes * num_classes
    Atr = np.array(A).transpose(0, 2, 1, 3).reshape(dim0, dim0)
    pre_eye = np.eye(num_classes, dtype = int)
    
    for pi in range(num_states):
        repr_str = np.base_repr(pi, base = num_classes, padding = num_nodes)[-num_nodes:]
        y = np.array(list(repr_str), dtype = int)
        x = pre_eye[y].reshape(dim0, 1)
        probs[pi] = np.exp(x.T.dot(Atr).dot(x) / 2).item()
    
    probs /= np.sum(probs)

    return probs


# sampling from a distribution with exact probabilities computed
def sampling(num_nodes, num_classes, num_samples, probs):

    num_states = probs.size
    samples = np.zeros((num_samples, num_nodes), dtype = int)
    index_samples = np.random.choice(np.arange(num_states), size = num_samples, p = probs)
    for si in range(num_samples):
        repr_str = np.base_repr(index_samples[si], base = num_classes, padding = num_nodes)[-num_nodes:]
        samples[si] = np.array(list(repr_str), dtype = int)
    
    return samples


# add noise to data
def add_noise(samples, num_classes, p_noise, noise_model):

    assert 0 <= p_noise and p_noise <= 1
    assert noise_model in ['noiseless', 'huber', 'independent']
    
    num_samples, num_nodes = samples.shape

    if noise_model == 'noiseless':
        pass
    elif noise_model == 'huber':
        W = np.random.binomial(1, p_noise, num_samples).astype(bool)
        samples[W] = np.random.randint(num_classes, size = (W.sum(), num_nodes))
    elif noise_model == 'independent':
        W = np.random.binomial(1, p_noise, samples.shape).astype(bool)
        sel_entries = samples[W]
        r = np.random.randint(num_classes - 1, size = sel_entries.shape)
        r += (r >= sel_entries).astype(int)
        samples[W] = r

    return samples


# get a diamond-shaped graph, returning weight and adjacency matrix
def get_diamond_graph(num_nodes, num_classes, theta):

    A = np.zeros((num_nodes, num_nodes, num_classes, num_classes))
    G = np.zeros((num_nodes, num_nodes), dtype = bool)

    cW = np.ones((num_classes, num_classes)) * theta
    cW[0::2, 1::2] = -theta
    cW[1::2, 0::2] = -theta

    A[0, 1:-1] = A[-1, 1:-1] = A[1:-1, 0] = A[1:-1, -1] = cW
    G[0, 1:-1] = G[-1, 1:-1] = G[1:-1, 0] = G[1:-1, -1] = True
    fW = np.random.randint(2, size = (num_nodes - 2, 1, 1)) * 2 - 1
    A[0, 1:-1] *= fW
    A[1:-1, 0] *= fW
    fW = np.random.randint(2, size = (num_nodes - 2, 1, 1)) * 2 - 1
    A[-1, 1:-1] *= fW
    A[1:-1, -1] *= fW

    return A, G


# get a grid-shaped graph, returning weight and adjacency matrix
def get_grid_graph(num_nodes, num_classes, theta):

    A = np.zeros((num_nodes, num_nodes, num_classes, num_classes))
    G = np.zeros((num_nodes, num_nodes), dtype = bool)

    cW = np.ones((num_classes, num_classes)) * theta
    cW[0::2, 1::2] = -theta
    cW[1::2, 0::2] = -theta

    ne = np.sqrt(num_nodes).astype(int)
    for i in range(ne):
        for j in range(ne):
            idx = i * ne + j
            if j > 0:
                G[idx, idx - 1] = G[idx - 1, idx] = True
                if np.random.rand() < 0.5:
                    A[idx, idx - 1] = A[idx - 1, idx] = cW
                else:
                    A[idx, idx - 1] = A[idx - 1, idx] = -cW
            if i > 0:
                G[idx, idx - ne] = G[idx - ne, idx] = True
                if np.random.rand() < 0.5:
                    A[idx, idx - ne] = A[idx - ne, idx] = cW
                else:
                    A[idx, idx - ne] = A[idx - ne, idx] = -cW

    return A, G


# get a random graph, returning weight and adjacency matrix
def get_random_graph(num_nodes, num_classes, theta):

    A = np.zeros((num_nodes, num_nodes, num_classes, num_classes))
    G = np.zeros((num_nodes, num_nodes), dtype = bool)

    cW = np.ones((num_classes, num_classes)) * theta
    cW[0::2, 1::2] = -theta
    cW[1::2, 0::2] = -theta

    for i, j  in itertools.combinations(range(num_nodes), 2):
        G[i, j] = G[j, i] = (np.random.rand() > 0.5)
        A[i, j] = A[j, i] = G[i, j] * (np.random.randint(2) * 2 - 1) * cW

    return A, G


# generate samples from a type of graph and perturb the data
def generate_samples(num_nodes, num_classes, graph_type, theta, num_samples, p_noise, noise_model):

    assert graph_type in ['diamond', 'grid', 'random']
    if graph_type == 'diamond':
        A, G = get_diamond_graph(num_nodes, num_classes, theta)
    elif graph_type == 'grid':
        A, G = get_grid_graph(num_nodes, num_classes, theta)
    elif graph_type == 'random':
        A, G = get_random_graph(num_nodes, num_classes, theta)

    probs = compute_probabilities(num_nodes, num_classes, A)
    samples = sampling(num_nodes, num_classes, num_samples, probs)
    samples = add_noise(samples, num_classes, p_noise, noise_model)

    return A, G, samples


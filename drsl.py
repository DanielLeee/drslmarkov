from scipy.optimize import minimize
import numpy as np
import itertools
import dataset


# log(sum_x exp(x))
def log_exp_sum(x):

    return x.max() + np.log(np.sum(np.exp(x - x.max())))


# log(1+exp(x))
def log_exp(x):

    ret = np.array(x)
    g0 = (x > 0)
    ret[g0] = x[g0] + np.log(1 + np.exp(-x[g0]))
    ret[~g0] = np.log(1 + np.exp(x[~g0]))

    return ret


# 1/(1+exp(x))
def recp_exp(x):
    
    ret = np.array(x)
    g0 = (x > 0)
    ret[g0] = np.exp(-x[g0]) / (1 + np.exp(-x[g0]))
    ret[~g0] = 1 / (1 + np.exp(x[~g0]))

    return ret


# objective and gradient for kl dro
def drlr_kl_func_grad(params, samples, epsilon):

    num_samples, num_nodes = samples.shape

    params = np.array(params)
    alpha = params[0]
    w = params[1:].reshape(num_nodes, -1)
    num_classes = w.shape[1]

    x_hat = np.concatenate((samples[:, :-1], np.zeros((num_samples, 1), dtype = int)), axis = 1)
    y_hat = np.array(samples[:, -1])
    ywx = y_hat * w[np.tile(np.arange(num_nodes), (num_samples, 1)), x_hat].sum(1)
    yx = np.expand_dims(y_hat, axis = (1, 2)) * np.eye(num_classes, num_classes)[x_hat]
    lywx = log_exp(-ywx)
    ln_M = np.log(1 / num_samples) + log_exp_sum(lywx / alpha)
    lywx_alpha_neg_exp = np.exp((lywx / alpha) - (lywx / alpha).max())
    lywx_alpha_exp_probs = lywx_alpha_neg_exp / lywx_alpha_neg_exp.sum()

    d_alpha = ln_M + alpha * (lywx_alpha_exp_probs * lywx / (-alpha * alpha)).sum() + epsilon
    d_w = (np.expand_dims(lywx_alpha_exp_probs * recp_exp(ywx), axis = (1, 2)) * (-yx)).sum(0)

    obj_val = alpha * ln_M + alpha * epsilon
    grads = np.concatenate((np.array([d_alpha]), d_w.reshape(-1)))

    return [obj_val, grads]


# objective and gradient for wasserstein dro
def drlr_func_grad(params, samples, epsilon, kappa):

    num_samples, num_nodes = samples.shape

    params = np.array(params)
    beta = params[0]
    w = params[1:].reshape(num_nodes, -1)
    num_classes = w.shape[1]
    
    x_hat = np.concatenate((samples[:, :-1], np.zeros((num_samples, 1), dtype = int)), axis = 1)
    y_hat = np.array(samples[:, -1])
    bat_opt_val = np.ones(num_samples) * float('-inf')
    bat_d_beta = np.zeros(num_samples)
    bat_d_w = np.zeros(([num_samples] + list(w.shape)))
    
    for g in [-1, 1]:
        cur_x_hat = np.array(x_hat)
        base_w = g * w[np.tile(np.arange(num_nodes), (num_samples, 1)), x_hat]
        dt_w = np.tile((g * w[:-1]).max(1), (num_samples, 1)) - base_w[:, :-1]
        w_sorted_rows = np.argsort(dt_w, axis = 1)[:, ::-1]
        w_sorted_cols = (g * w[:-1]).argmax(1)[w_sorted_rows]
        cur_exp_val = base_w.sum(1)
        for wi in range(-1, num_nodes - 1):
            if wi > -1:
                cur_exp_val += dt_w[np.arange(num_samples), w_sorted_rows[:, wi]]
                cur_x_hat[np.arange(num_samples), w_sorted_rows[:, wi]] = w_sorted_cols[:, wi]
            bat_cur_val = log_exp(cur_exp_val) - 2 * beta * (wi + 1) - 0.5 * beta * kappa * (1 + g * y_hat)
            opt_flag = (bat_cur_val > bat_opt_val)
            bat_opt_val[opt_flag] = bat_cur_val[opt_flag]
            bat_d_beta[opt_flag] = -2 * (wi + 1) - 0.5 * kappa * (1 + g * y_hat[opt_flag])
            bat_d_w[opt_flag] = g * np.expand_dims(recp_exp(-cur_exp_val[opt_flag]), axis = (1, 2)) * np.eye(num_classes, num_classes)[cur_x_hat[opt_flag]]
    
    obj_val = beta * epsilon + bat_opt_val.mean()
    d_beta = epsilon +  bat_d_beta.mean()
    d_w = bat_d_w.mean(0)
    
    grads = np.concatenate((np.array([d_beta]), d_w.reshape(-1)))

    return [obj_val, grads]


# structure learning by wasserstein or kl dro
def graph_learn(num_runs, num_nodes, num_classes, graph_type, min_weight, num_samples, epsilon, kappa, p_noise, noise_model, ambiguity_set):

    assert ambiguity_set in ['kl', 'wass']

    print('#runs = {}, #nodes = {}, #classes = {}, #samples = {}, epsilon = {}, kappa = {}, prob_noise = {}, noise_type = {}, ambiguity_set = {}, graph_type = {}, min_weight = {}'.format(num_runs, num_nodes, num_classes, num_samples, epsilon, kappa, p_noise, noise_model, ambiguity_set, graph_type, min_weight))

    cnt_success = 0
    for ri in range(num_runs):
        A_gt, G_gt, samples = dataset.generate_samples(num_nodes = num_nodes, num_classes = num_classes, graph_type = graph_type, theta = min_weight, num_samples = num_samples, p_noise = p_noise, noise_model = noise_model)
        A_hat = np.zeros((num_nodes, num_nodes, num_classes, num_classes))
        for ni in range(num_nodes):
            for class0, class1 in itertools.combinations(range(num_classes), 2):
                cur_samples = samples[(samples[:, ni] == class0) | (samples[:, ni] == class1)]
                cur_samples[cur_samples[:, ni] == class1, ni] = -1
                cur_samples[cur_samples[:, ni] == class0, ni] = 1
                cur_samples = np.concatenate((cur_samples[:, :ni], cur_samples[:, ni+1:], cur_samples[:, ni:ni+1]), axis = 1)
                initial_params = np.random.rand(num_nodes * num_classes + 1)
                bnds = tuple([(1e-9, None)] + [(None, None) for i in range(num_nodes * num_classes)])
                if ambiguity_set == 'wass':
                    opt_func_grad = drlr_func_grad
                    args_tup = (cur_samples, epsilon / np.sqrt(num_samples), kappa)
                elif ambiguity_set == 'kl':
                    opt_func_grad = drlr_kl_func_grad
                    args_tup = (cur_samples, epsilon / num_samples)
                optimal_params = minimize(opt_func_grad, initial_params, args = args_tup, method = 'L-BFGS-B', jac = True, bounds = bnds, options = {'disp' : 0})
                optimal_params = np.array(optimal_params.x)
                optimal_w = optimal_params[1:].reshape(num_nodes, num_classes)
                optimal_w[:-1] -= optimal_w[:-1].mean(1, keepdims = True)
                optimal_w = np.concatenate((optimal_w[:ni], np.zeros((1, num_classes)), optimal_w[ni:-1]), axis = 0)
                A_hat[ni, :, class0, :] += optimal_w / num_classes
                A_hat[ni, :, class1, :] -= optimal_w / num_classes
        G_hat = np.abs(A_hat).max(axis = (2, 3)) > min_weight / 2
        # print((A_hat - A_gt).max())
        # print((G_hat != G_gt).sum())
        if np.all(G_gt == G_hat):
            cnt_success += 1
    
    print('Successful rate: {} / {}'.format(cnt_success, num_runs))
    
    return 0



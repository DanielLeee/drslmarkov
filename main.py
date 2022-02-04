import numpy as np
import time
import drsl


# main function for all the dro experiments
def main():

    num_runs = 100
    num_nodes = 6
    num_classes = 4
    graph_type = 'random' # grid | diamond | random
    min_weight = 0.2
    p_noise = 0.2
    noise_model = 'noiseless' # noiseless | huber | independent
    epsilon = 1.2 # ambiguity
    kappa = 1.0 # metric
    ambiguity_set = 'wass' # kl | wass
    
    for num_samples in np.arange(1000, 10001, 1000):
        start_time = time.time()
        drsl.graph_learn(num_runs = num_runs, num_nodes = num_nodes, num_classes = num_classes, graph_type = graph_type, min_weight = min_weight, num_samples = num_samples, epsilon = epsilon, kappa = kappa, p_noise = p_noise, noise_model = noise_model, ambiguity_set = ambiguity_set)
        print('Elapsed time: {}'.format(time.time() - start_time))

    return 0


if __name__ == '__main__':

    main()


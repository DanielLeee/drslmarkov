# Distributionally Robust Structure Learning for Discrete Pairwise Markov Networks

This is the official implementation of the following paper accepted to *AISTATS 2022*:

> **Distributionally Robust Structure Learning for Discrete Pairwise Markov Networks**
> 
> Yeshu Li, Zhan Shi, Xinhua Zhang, Brian D. Ziebart
> 
> [[PMLR link]](https://proceedings.mlr.press/v151/li22f.html)

## Requirements

- numpy
- scipy

## Quick Start

Run

```shell
python main.py
```

## Citation

Please cite our work if you find it useful in your research:

```
@InProceedings{pmlr-v151-li22f,
  title = 	 { Distributionally Robust Structure Learning for Discrete Pairwise Markov Networks },
  author =       {Li, Yeshu and Shi, Zhan and Zhang, Xinhua and Ziebart, Brian},
  booktitle = 	 {Proceedings of The 25th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {8997--9016},
  year = 	 {2022},
  editor = 	 {Camps-Valls, Gustau and Ruiz, Francisco J. R. and Valera, Isabel},
  volume = 	 {151},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {28--30 Mar},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v151/li22f/li22f.pdf},
  url = 	 {https://proceedings.mlr.press/v151/li22f.html},
  abstract = 	 { We consider the problem of learning the underlying structure of a general discrete pairwise Markov network. Existing approaches that rely on empirical risk minimization may perform poorly in settings with noisy or scarce data. To overcome these limitations, we propose a computationally efficient and robust learning method for this problem with near-optimal sample complexities. Our approach builds upon distributionally robust optimization (DRO) and maximum conditional log-likelihood. The proposed DRO estimator minimizes the worst-case risk over an ambiguity set of adversarial distributions within bounded transport cost or f-divergence of the empirical data distribution. We show that the primal minimax learning problem can be efficiently solved by leveraging sufficient statistics and greedy maximization in the ostensibly intractable dual formulation. Based on DRO’s approximation to Lipschitz and variance regularization, we derive near-optimal sample complexities matching existing results. Extensive empirical evidence with different corruption models corroborates the effectiveness of the proposed methods. }
}
```

## Acknowledgement

This project is based upon work supported by the National Science Foundation under Grant No. 1652530.

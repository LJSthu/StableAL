## StableAL
Code for our paper Stable Adversarial Learning under Distributional Shifts (AAAI 2021)

Here we release the code of our SAL algorithm as well as the spurious correlation experiment.
Code for other experiments will be available soon.

#### Spurious Correlation Experiment Details
Here are some tunable parameters in the data generation, including:
* $p$: the dimension of $X$
* $pvb$: the dimension of the spurious attribute
* $n_2$: the number of training data points for environment one
* $n_3$: the number of training data points for the other environments
* $r$: the $r$ of the major training environment
* r_list: the $r$ of the other environments


#### Stable Adversarial Algorithm Details
Here are some tunable parameters in our algorithm, including:
* `epoch`: the number of the whole iterations
* `epoch_theta`: the number of iterations for training model parameters $\theta$
* `epoch_attack`: the number of iteration for updating the adversarial samples $X_A$

Notably that in the last epoch, we fix the learned covariate weights and train the model parameters for 5000 epochs. (in function `trainAll`)


ps: We find our algorithm is sensitive to the choices of hyper-parameters, which need to be tuned carefully.

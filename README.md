# BIP: *B*ayesian *I*nverse *P*roblems

BIP is a Python library providing a toolbox for solving inverse problems using probabilistic methods. 
What typically differentiates these problems from those addressed by standard probabilistic programming languages
(e.g., PyMC and Stan) is the presence of a computationally expensive forward model underlying the likelihood function.
Evaluating these forward model at a particular parameter setting usually implies running a numerical ODE or PDE solver.
The goal in such settings is to approximate the posterior distribution over the forward model parameters using as few 
forward model evaluations as possible. 

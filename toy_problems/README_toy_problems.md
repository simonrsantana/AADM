
# Toy problems codes

In both of these folders there are the codes and data used in the synthetic examples for data with bimodal and heteroscedastic posterior distributions. 

The structure of the system is as described in the paper. Moreover, to check the effects of pre-training and to avoid the possibility of the system falling into local minima, both codes are set up so a "warming-up" period can be performed. During this period, the system can train using a given value of _alpha_, which changes afterwards to the second and final given value. This allows for training the algorithm with a high value for _alpha_ such as 1.0 to then change it to whatever value we are interested in thus avoiding fitting the system to a suboptimal value. 


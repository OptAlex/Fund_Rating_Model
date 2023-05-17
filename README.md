Set up the project: 
- We recommend setting the source folder to the overall project (marketrisk) to use the same paths as we do. Otherwise paths might need to be adjusted.
- To install all required packages we suggest using a virtual environment and run pip install -r requirements.txt
- To speed up debugging we included a ETF_List_short.xlsx which significantly decreases runtime

What is the project about?

The Topic of this project is to calculate the default probabilities and CVaR of a portfolio of funds using an ARMA-GARCH + Copula model to simulate 252 daily returns for each fund 1000 times in R. After the simulation, the default probability and CVaR is calculated in Python. All relevant help functions can be found in the folder help_functions. The main Process can be found in main.py. We are using parallelization, so runtime is depending on the number of CPU cores available. Because of numeric issues (see comment about try and except in the main.py) we figured out, that the most stable way to run the script while still using significantly large number of bootstrapping and simulations is 10x bootstrapping with 1000 simulation for 26 funds and 252 days. by itterating trough the for loop in the main.py we were able to gather 580 individual bootstrapping and perform a full simulation run on each of them. Plotting and analysing them was done separately and is not automated.

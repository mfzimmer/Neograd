# Neograd

This repo supports the paper 
<a href="http://www.arxiv.org/pdf/2010.07873.pdf">Neograd: Gradient Descent with a Near-Ideal Learning Rate</a>,
which introduces the algorithm "Neograd".
The paper and associated code are by Michael F. Zimmer.
It's been submitted to JMLR.




## Getting Started

Download the code.  Paths within the program are relative.


### Prerequisites

Python 3<BR>
Jupyter notebook


### Installing

Unzip/clone the repo.  You should see this directory structure:<BR>
neograd/<BR>
	libs/<BR>
    notebooks/<BR>
    figs/<BR>
The meaning of these names is self-explanatory.  Only the name "notebooks" can be changed without interfering with the paths.

    
### Running Notebooks

After cd-ing into the "notebooks" directory, open a notebook in Jupyter and execute the cells.  If you choose to uncomment certain lines (the save fig command) a figure will be saved for you.  Some of these are the same figs that appear in the aforementioned paper.



### Descriptions of notebooks

These experiment notebooks contain evaluations of algorithms against the named cost fcn<BR>
EXPT\_2Dshell<BR>
EXPT\_Beale<BR>
EXPT\_double<BR>
EXPT\_quartic<BR>
EXPT\_sigmoid-well<BR>

Additionally, these contain additional tests.<BR>
EXPT\_hybrid<BR>
EXPT\_manual<BR>
EXPT\_momentum<BR>



### Descriptions of libraries

<B>algos\_vec</B><BR>
Functions that are central to the GD family and Neograd family.


<B>common</B><BR>
Functions for rho, alpha, and functions for tracking results of a run.


<B>common\_vec</B><BR>
Functions used by algos_vec, which aren't central to the algorithms.
Also, these functions have a specific assumption that the "parameter vector" is a numpy array.


<B>costgrad\_vec</B><BR>
This is an aggregation of all the functions needed to compute the cost and gradient of the specific cost functions examined in the paper.


<B>params</B><BR>
Contains all global parameters (not to be confused with the parameter vector that is being optimized).  Also present is a function to return a "good choice" of alpha for each algorithm-cost function combination, as determined by trial and error.


<B>plotting</B><BR>
The plotting functions are passed the dictionaries of results returned by the optimization runs


### A few details

"p" represents the parameter vector in the repo; note this differs from "theta" which is used in the paper.

Statistics during the run are accumulated by a dictionary of lists.
The keys in the dictionary contain the name of the statistic, and the "values" are lists.
Before entering the main loop, the names/keys must be declared; this is done in the function "init\_results".  After each iteration, a list will have a value appended to it; this is done in the function "update\_results".  Both of these functions are in the "common" library.

If you set the total iteration number ("num") too high, you may find you get underflow errors plus their ramifications.  This is because the Neograd algorithm will drive the error down to be so small, it bumps up against machine precision.  There are a number of sophisticated ways to handle this, but for the purposes here it is enough to simply stop the optimization before it becomes an issue.

In the code on github, this alternative definition of rho may be used.
Simply change the parameter "g\_rhotype" to "original", instead of "new".
This is discussed in an appendix of the paper.





## Author

Michael F. Zimmer



## License

This project is licensed under the MIT license.






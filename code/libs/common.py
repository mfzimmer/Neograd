
import numpy as np


# ====================================================================
def get_rho(cost_old, cost_new, est_diff, rho_type):
    if rho_type == "original":
        return( get_rho_original(cost_old, cost_new, est_diff) )
    elif rho_type == "new":
        return( get_rho_new(cost_old, cost_new, est_diff) )
    else:
        print("ERROR in get_rho, rho_type=", rho_type)
        return( None)


# ====================================================================
def get_rho_new(cost_old, cost_new, est_diff):
    try:
        cost_est = cost_old + est_diff
        rho = np.abs( (cost_new - cost_est) / (cost_old - cost_est) )
#        rho = np.abs( (cost_new - cost_old - est_diff) / (cost_old - cost_old - est_diff) )
    except ZeroDivisionError:
        print("ERROR in get_rho: ZeroDivisionError")
    return( rho )

# ====================================================================
def get_rho_original(cost_old, cost_new, est_diff):
    try:
        cost_est = cost_old + est_diff
        rho = np.abs( (cost_new - cost_est) / (cost_new - cost_old) )
#        rho = np.abs( (cost_new - cost_old - est_diff) / (cost_new - cost_old) )
    except ZeroDivisionError:
        print("ERROR in get_rho: ZeroDivisionError")
    return( rho )


# ====================================================================
def get_rho_error( rho, rho_basis ):
    # rho_basis = rho that's used for basis of comparison
    rel_error = (rho - rho_basis) / rho_basis
    return( rel_error )


# ====================================================================
def get_alpha_prime_0( A, B, rho_, rho_type ):
    if rho_type == "original":
        return( get_alpha_prime_0_original( A, B, rho_ ) )
    elif rho_type == "new":
        return( get_alpha_prime_0_new( A, B, rho_ ) )
    else:
        print("ERROR in get_alpha_prime_0, rho_type=", rho_type)
        return( None)
        
# ====================================================================
def get_alpha_prime_0_new( A, B, rho_ ):
    alpha_prime = np.abs(B/A) * rho_
    return( alpha_prime )

# ====================================================================
def get_alpha_prime_0_original( A, B, rho_ ):
    alpha_prime = (B/A) * rho_ / (1 - rho_)  #alpha_plus
    if alpha_prime < 0:
        alpha_prime = -(B/A) * rho_ / (1 + rho_)  #alpha_minus
    if alpha_prime < 0:
        print("ERROR in get_alpha_prime_0")
    return( alpha_prime )

# ====================================================================
def get_rho_prime( rho, rho_targ ):
    # heuristic
    if rho < rho_targ:
        r = 0.75*np.log10(rho/rho_targ)
        rho_prime = np.power( 10, r ) * rho_targ
    else:
        rho_prime = rho_targ
#        rho_prime = rho_targ * np.sqrt(rho_targ/rho)
#        rho_prime = (0.015 + rho_targ)/2.0
#        rho_prime = rho_targ / (1.0 + np.log10(rho/rho_targ) )
#        print(" scaled rho_prime = ", rho_prime)
    return( rho_prime )

# ====================================================================
#def get_rho_AB(alpha,A,B):
#    rho_ = np.abs( alpha*A/(alpha*A + B) )
#    return( rho_ )


# ====================================================================
# Return a dictionary of empty lists, which will track stats
def init_results(stats_):
    res_ = {}
    for c in stats_:
        res_[c] = []
    return(res_) 


# ====================================================================
# Use the key-value pairs in d_ to update the stats in the dictionary res_.
# The keys in d_ are a subset of the keys in res_. (ie, not necessarilly a proper subset)
def update_results(res_, d_):
    # TO DO: check that d_.keys() is in res_.keys()
    for k in d_.keys():
        res_[k].append( d_[k])

        
# ====================================================================
def do_printout(d_, ix, nfreq):
    if ix % nfreq == 0 and ix > 0:
        print("{}: cost,rho,dotp = {}, {}, {}".format( ix, d_["cost"][-1],  d_["rho"][-1], d_["dotp"][-1] ) )



# ====================================================================
# Returns (n,nc,m) 
def get_dims(X,Y):
    print("[X] := (nx,m)")
    print("[Y] := (ny,m)")
    
    (nx,m) = X.shape
    (ny,m2) = Y.shape
    
    print("nx  = ", nx)
    print("ny  = ", ny)
    print("m   = ", m)

    if m != m2:
        print("ERROR, m != m2.  Different number of records in X and Y")

    return(nx,ny,m)



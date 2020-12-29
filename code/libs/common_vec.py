
import numpy as np
import params as pm


# ====================================================================
def get_p_new(p1,p2):
    return(p1 + p2)

# ====================================================================
def get_est_diff(grad, dp):
    return( np.dot(grad,dp) )

# ====================================================================
def get_dotp(dp1, dp2):
    try:
        dotp = np.dot(dp1,dp2) / np.sqrt(np.dot(dp1,dp1) * np.dot(dp2,dp2))
#    except ZeroDivisionError:
    except ArithmeticError:
        print("ERROR in get_dotp: divide by zero")
        dotp = 1.0

    return(dotp)

# ====================================================================
def get_dotp_2(dp1, dp2):
    if (dp1 is None) or (dp2 is None):
        return None
    else:
        return( get_dotp(dp1,dp2) )

# ====================================================================
#def get_dotp_3(dp1, dp2, i):
#    if i == 0:
#        return(1.0)
#    else:
#        return( get_dotp(dp1,dp2) )
    
    

# ====================================================================
def get_sumd_x(dp):
    return( np.sqrt(np.dot(dp,dp)) )

# ====================================================================
def get_sumd_y(p1,p2):
    return( np.sqrt(np.dot(p1-p2,p1-p2)) )


# ====================================================================
# Use this to measure the distance from p_init to p_target
def get_dist(p1, p2):
    return( np.sqrt(np.dot(p1-p2,p1-p2)) )


# ====================================================================
def get_step_size(dp):
    return( np.sqrt(np.dot(dp,dp)) )


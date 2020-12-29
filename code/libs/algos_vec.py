
import numpy as np
import costgrad_vec as cg 
import common as co
import params as pm
import common_vec as cv


# ====================================================================
def get_dp_momentum( beta, grad, v, i, alpha ):
    v     = beta * v + (1-beta) * grad
    v_est = v / (1 - np.power(beta, i+1))
    dp    = -alpha * v_est
    return(dp,v)


# ====================================================================
def get_dp_Nesterov( name, p, beta, grad, v, alpha ):
    v     = beta * v + alpha * cg.get_grad(name, p -beta*v)
    dp    = -v
    return(dp,v)


# ====================================================================
def get_dp_RMS( eps, beta2, grad, v2, i, alpha ):
    v2     = beta2 * v2 + (1-beta2) * grad*grad   #element-wise product
    v2_est = v2 / (1 - np.power(beta2, i+1))
    temp   =  grad / np.sqrt(v2_est + eps)
    dp     = -alpha * temp
    return(dp,v2)


# ====================================================================
def get_dp_Adam( eps, beta, beta2, grad, v, v2, i, alpha ):
    v        = beta * v + (1-beta) * grad
    v2       = beta2 * v2 + (1-beta2) * grad*grad   #element-wise product
    v_est    = v  / (1 - np.power(beta, i+1))
    v2_est   = v2 / (1 - np.power(beta2, i+1))
    temp     = v_est / np.sqrt(v2_est + eps)
    dp       = -alpha * temp
    return(dp,v,v2)


# ====================================================================
# This allows the use of any of these optimization algos according to parameter "type_opt"
def get_dp_choice(type_opt, grad, p_old, v, v2, i, alpha):
    eps   = pm.g_eps_adam
    beta  = pm.g_beta
    beta2 = pm.g_beta2
    
    if type_opt   == "dp_GD_basic":
        dp      = -alpha * grad
    elif type_opt == "dp_GD_momentum":
        dp,v    = get_dp_momentum( beta, grad, v, i, alpha )
    elif type_opt == "dp_GD_Nesterov":
        dp,v    = get_dp_Nesterov( name, p_old, beta, grad, v, alpha )
    elif type_opt == "dp_RMSProp":
        dp,v2   = get_dp_RMS( eps, beta2, grad, v2, i, alpha )
    elif type_opt == "dp_Adam":
        dp,v,v2 = get_dp_Adam( eps, beta, beta2, grad, v, v2, i, alpha )
    else:
        print("ERROR in get_dp_choice.  type_opt=", type_opt)
        return(None,None)
    
    return( dp, v, v2 )


# ====================================================================
# This is the algorithm to run gradient descent (GD), whether it be plain GD, GD with momentum, Adam, etc
def do_GDFamily(name, alpha, num, p_old, p_target, type_opt, rho_targ, b_print, i_manual, i_factor, rho_type):  

    v     = np.zeros(p_old.shape, dtype=np.float128)
    v2    = np.zeros(p_old.shape, dtype=np.float128)

    # init ---------------------------
    p_ref     = p_old.copy()  #a copy of initial p, to compute sumd_y
    cost_old  = cg.get_cost(name, p_old)
    print("cost_old = ", cost_old)
    dp_old    = None
    dist      = cv.get_dist(p_old, p_target)
    d_res     = co.init_results( ["cost","rho","dotp","sumd_x","sumd_y","p","rho_error","dist"] )
    co.update_results( d_res, {"p":p_old.copy(), "cost":cost_old, "sumd_x":0.0, "sumd_y":0.0, "dist":dist} )
    sumd_x   = 0
    
    for i in range(num):
        
        # Extra booast at i==i_manual ----------------
        if i == i_manual:
            alpha *= i_factor
            
        # grad -------------------------
        grad      = cg.get_grad(name, p_old)

        # optimize ---------------------
        dp,v,v2   = get_dp_choice(type_opt, grad, p_old, v, v2, i, alpha)
        
        # cost -------------------------
        p_new     = cv.get_p_new(p_old, dp)
        cost_new  = cg.get_cost(name, p_new)

        # diagnostics ------------------
        est_diff  = cv.get_est_diff(grad, dp)
        rho       = co.get_rho(cost_old, cost_new, est_diff, rho_type)
        rho_error = co.get_rho_error( rho, rho_targ )
        dotp      = cv.get_dotp_2(dp_old, dp)
        sumd_x   += cv.get_sumd_x(dp)  #a running total
        sumd_y    = cv.get_sumd_y(p_ref, p_new)  #a point measurement

        # updates ----------------------
        p_old    = p_new.copy()
        cost_old = cost_new
        dp_old   = dp.copy()
        dist     = cv.get_dist(p_old, p_target)
        co.update_results(d_res, {"rho":rho, "p":p_old.copy(), "cost":cost_old, "dotp":dotp, "sumd_x":sumd_x, "sumd_y":sumd_y, "rho_error":rho_error, "dist":dist} )

        # printout ---------------------
        if(b_print):
            co.do_printout(d_res, i, nfreq=1)
    
    return(d_res)



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# ====================================================================
# The list of input arguments can be cleaned up.  Eg, v and v2 are no longer needed, and type_opt is fixed,
# so that can be dropped too.  Also, even the call to "get_dp_choice" can be replaced by Basic GD version 
# (although this isn't a big deal).  Also, pass in value for n_rep.  Maybe also for rhomin & rhomax,
# in case I want to create a conditional based on those.
def get_starting_alpha(name, p_old, type_opt, v, v2, rho_targ, cost_old, rho_type, alpha = 0.000001):
    cost_old  = cg.get_cost(name, p_old)
    grad      = cg.get_grad(name, p_old)
    # Turn these two params into input parameters
    n_rep = 20
    alpha_last = None
    rho_last   = None
    
    for j in range(n_rep):
        # optimize ---------------------
        type_opt = "dp_GD_basic"
        i = 0
        dp,v_temp,v2_temp = get_dp_choice(type_opt, grad, p_old, v, v2, i, alpha)

        # cost -------------------------
        p_new     = cv.get_p_new(p_old, dp)
        cost_new  = cg.get_cost(name, p_new)
        est_diff  = cv.get_est_diff(grad, dp)

        # new stuff --------------------
        rho  = co.get_rho(cost_old, cost_new, est_diff, rho_type)
        rerr = (rho - rho_targ) / rho_targ
        print(j, ": alpha,rho,rerr = ", alpha,rho,rerr)
                
        if rho > 0.0:
            if rho_last is None:
                rho_last = rho
                alpha_last = alpha
            elif rho <= rho_targ and rho > rho_last:
                rho_last = rho
                alpha_last = alpha
                
        if rho > rho_targ:
            return(alpha_last) #it's possible this is None
            
            
        if rho == 0.0:  #Do this because sometimes it gets rounded down to 0.0
            alpha *= 2
        else:
            cost_est  = cost_old + est_diff
            A = (cost_new - cost_est)/(alpha*alpha)
            B = (cost_est - cost_old)/alpha
            rho_prime = co.get_rho_prime( rho, rho_targ )
            alpha     = co.get_alpha_prime_0( A, B, rho_prime, rho_type )
            
    return(alpha)
    

# ====================================================================
def do_Neograd(name, alpha, num, p_old, p_target, type_opt, rho_targ, b_print, rho_type):  
    
    # These variables are used as needed, according to value of 'type_opt'
    v     = np.zeros(p_old.shape, dtype=np.float128)
    v2    = np.zeros(p_old.shape, dtype=np.float128)

    # init ---------------------------
    p_ref     = p_old.copy()  #a copy of initial p, to compute sumd_y
    cost_old  = cg.get_cost(name, p_old)
#    print("cost_old = ", cost_old)
    dp_old    = None
    dist      = cv.get_dist(p_old, p_target)
    sumd_x    = 0
    rho_prime = rho_targ

    # Get a starting alpha if unspecified -----------------
    if alpha is None:
        alpha = get_starting_alpha(name, p_old, type_opt, v, v2, rho_targ, cost_old, rho_type)
        print("starting alpha = ", alpha)
    alpha_hat = alpha

    d_res     = co.init_results( ["cost","rho","dotp","sumd_x","sumd_y","p","alpha","rho_error","dist","step"] ) 
    co.update_results( d_res, {"p":p_old.copy(), "cost":cost_old, "sumd_x":0.0, "sumd_y":0.0, "alpha":alpha, "dist":dist} )

    for i in range(num):
        print(i, " ============================================")
        
        # grad -------------------------
        grad      = cg.get_grad(name, p_old)

        # optimize ---------------------
        dp,v,v2   = get_dp_choice(type_opt, grad, p_old, v, v2, i, alpha)
        
        # cost -------------------------
        p_new     = cv.get_p_new(p_old, dp)
        cost_new  = cg.get_cost(name, p_new)
        est_diff  = cv.get_est_diff(grad, dp)

        # diagnostics ------------------
        rho       = co.get_rho(cost_old, cost_new, est_diff, rho_type)
        rho_error = co.get_rho_error( rho, rho_targ )
        dotp      = cv.get_dotp_2(dp_old, dp)
        sumd_x   += cv.get_sumd_x(dp)  #a running total
        sumd_y    = cv.get_sumd_y(p_ref, p_new)  #a point measurement
        
        # new stuff --------------------
        cost_est = cost_old + est_diff
        A = (cost_new - cost_est)/(alpha*alpha)
        B = (cost_est - cost_old)/alpha
#        print(" A = ", A)
#        print(" B = ", B)
        
        # updates ----------------------
        dist     = cv.get_dist(p_old, p_target)
        p_old    = p_new.copy()
        cost_old = cost_new
        dp_old   = dp.copy()
        step     = cv.get_step_size(dp_old)

        alpha_hat = alpha
        rho_prime = co.get_rho_prime( rho, rho_targ )
        alpha     = co.get_alpha_prime_0( A, B, rho_prime, rho_type )
#        alpha     = co.get_alpha_prime_0( A, B, rho_targ, rho_type )

        co.update_results(d_res, {"rho":rho, "p":p_old.copy(), "cost":cost_old, "dotp":dotp, "sumd_x":sumd_x, "sumd_y":sumd_y, "alpha":alpha_hat, "rho_error":rho_error, "dist":dist, "step":step} )

        # printout ---------------------
        print("alpha,rho = ", alpha, rho, " ---------------" )
        if(b_print):
            co.do_printout(d_res, i, nfreq=1)
    
    return(d_res)



# ====================================================================
def do_Neograd_dbl(name, alpha, num, p_old, p_target, type_opt, rho_targ, b_print, rho_type, n_rep=2):  
    
    # These variables are used as needed, according to value of 'type_opt'
    v     = np.zeros(p_old.shape, dtype=np.float128)
    v2    = np.zeros(p_old.shape, dtype=np.float128)

    # init ---------------------------
    p_ref     = p_old.copy()  #a copy of initial p, to compute sumd_y
    cost_old  = cg.get_cost(name, p_old)
#    print("cost_old = ", cost_old)
    dp_old    = None
    dist      = cv.get_dist(p_old, p_target)
    sumd_x   = 0
    rho_prime = rho_targ
    
    # Get a starting alpha if unspecified -----------------
    if alpha is None:
        alpha = get_starting_alpha(name, p_old, type_opt, v, v2, rho_targ, cost_old, rho_type)
        print("starting alpha = ", alpha)
    alpha_hat = alpha

    d_res     = co.init_results( ["cost","rho","dotp","sumd_x","sumd_y","p","alpha","rho_error","dist"] )
    co.update_results( d_res, {"p":p_old.copy(), "cost":cost_old, "sumd_x":0.0, "sumd_y":0.0, "alpha":alpha, "dist":dist} )

    for i in range(num):
        print(i, " ------------------------------")
        # grad -------------------------
        grad      = cg.get_grad(name, p_old)

        for j in range(n_rep):
#            print(j, " - - - - - - - - - - - - ")
            # optimize ---------------------
            dp,v_temp,v2_temp = get_dp_choice(type_opt, grad, p_old, v, v2, i, alpha)

            # cost -------------------------
            p_new     = cv.get_p_new(p_old, dp)
            cost_new  = cg.get_cost(name, p_new)
            est_diff  = cv.get_est_diff(grad, dp)

            # new stuff --------------------
            rho       = co.get_rho(cost_old, cost_new, est_diff, rho_type)
            print("   j,rho = ", j, rho)
            cost_est  = cost_old + est_diff
            A = (cost_new - cost_est)/(alpha*alpha)
            B = (cost_est - cost_old)/alpha
            alpha_hat = alpha
            rho_prime = co.get_rho_prime( rho, rho_targ )
            alpha     = co.get_alpha_prime_0( A, B, rho_prime, rho_type )
#            alpha     = co.get_alpha_prime_0( A, B, rho_targ, rho_type )
        v  = v_temp.copy()  #check for memory leaks with v,v2 and temps
        v2 = v2_temp.copy()

        # diagnostics ------------------
#        rho_error = co.get_rho_error( rho, rho_prime )
        rho_error = co.get_rho_error( rho, rho_targ )
        dotp      = cv.get_dotp_2(dp_old, dp)
        sumd_x   += cv.get_sumd_x(dp)  #a running total
        sumd_y    = cv.get_sumd_y(p_ref, p_new)  #a point measurement
        dist      = cv.get_dist(p_old, p_target)

        # updates ----------------------
        p_old    = p_new.copy()
        cost_old = cost_new
        dp_old   = dp.copy()
#        print("i,rho = ", i, rho)
        
        co.update_results(d_res, {"rho":rho, "p":p_old.copy(), "cost":cost_old, "dotp":dotp, "sumd_x":sumd_x, "sumd_y":sumd_y, "alpha":alpha_hat, "rho_error":rho_error, "dist":dist} )

        # printout ---------------------
        if(b_print):
            co.do_printout(d_res, i, nfreq=1)
    
    return(d_res)



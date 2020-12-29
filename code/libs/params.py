
# ================================================================
g_eps_adam = 10**(-8)
g_beta     = 0.9
g_beta2    = 0.999
g_rhomin   = 0.015
g_rhomax   = 0.15

#for 2Dshell
#g_ymin_2Dshell = None
#g_ymax_2Dshell = None
g_ymin_2Dshell = -7.9
g_ymax_2Dshell = 4.0

g_alpha_Neo_init = 0.01



# ================================================================
def get_alpha_num( name, type_opt ):
    if name == "x^2":
        if type_opt == "dp_GD_basic":
            alpha = 0.5
            num   = 100
        elif type_opt == "dp_GD_momentum":
            alpha = 0.01
            num   = 100
#        elif type_opt == "dp_GD_Nesterov":
#            alpha = 
#            num   = 
        elif type_opt == "dp_RMSProp":
            alpha = 0.1
            num   = 100
        elif type_opt == "dp_Adam":
            alpha = 0.03
            num   = 100
        else:
            print("ERROR in params.py: no match for type_opt with name == x^2.  type = ", type_opt)
    elif name == "x^4":
        if type_opt == "dp_GD_basic":
            alpha = 0.02
            num   = 200
        elif type_opt == "dp_GD_momentum":
            alpha = 0.004
            num   = 200
#        elif type_opt == "dp_GD_Nesterov":
#            alpha = 
#            num   = 
        elif type_opt == "dp_RMSProp":
            alpha = 0.5
            num   = 200
        elif type_opt == "dp_Adam":
            alpha = 0.15
            num   = 200
        else:
            print("ERROR in params.py: no match for type_opt with name == x^2.  type = ", type_opt)
    elif name == "ellipse":
        if type_opt == "dp_GD_basic":
            alpha = 0.2
            num   = 300
        elif type_opt == "dp_GD_momentum":
            alpha = 0.05
            num   = 300
#        elif type_opt == "dp_GD_Nesterov":
#            alpha = 
#            num   = 
        elif type_opt == "dp_RMSProp":
            alpha = 0.1
            num   = 300
        elif type_opt == "dp_Adam":
            alpha = 0.03
            num   = 300
        else:
            print("ERROR in params.py: no match for type_opt with name == x^2.  type = ", type_opt)
    elif name == "1Dsigwell":
        if type_opt == "dp_GD_basic":
            alpha = 0.2
            num   = 2000
        elif type_opt == "dp_GD_momentum":
            alpha = 0.1
            num   = 3000
#        elif type_opt == "dp_GD_Nesterov":
#            alpha = 
#            num   = 
        elif type_opt == "dp_RMSProp":
            alpha = 0.1
            num   = 500
        elif type_opt == "dp_Adam":
            alpha = 0.2
            num   = 500
        else:
            print("ERROR in params.py: no match for type_opt with name == x^2.  type = ", type_opt)
    elif name == "2Dshell":
        if type_opt == "dp_GD_basic":
            alpha = 0.1
            num   = 1200
        elif type_opt == "dp_GD_momentum":
            alpha = 0.7
            num   = 1200
#        elif type_opt == "dp_GD_Nesterov":
#            alpha = 
#            num   = 
        elif type_opt == "dp_RMSProp":
            alpha = 0.01
            num   = 1200
        elif type_opt == "dp_Adam":
            alpha = 0.15
            num   = 1200
        else:
            print("ERROR in params.py: no match for type_opt with name == x^2.  type = ", type_opt)
    elif name == "Beale":
        if type_opt == "dp_GD_basic":
            alpha = 0.00001
            num   = 400
        elif type_opt == "dp_GD_momentum":
            alpha = 0.00002
            num   = 400
#        elif type_opt == "dp_GD_Nesterov":
#            alpha = 
#            num   = 
        elif type_opt == "dp_RMSProp":
            alpha = 0.3
            num   = 400
        elif type_opt == "dp_Adam":
            alpha = 0.15
            num   = 400
        else:
            print("ERROR in params.py: no match for type_opt with name == x^2.  type = ", type_opt)
    else:
        print("ERROR in params.py: no match for name = ", name)

    return( alpha, num )
    
    

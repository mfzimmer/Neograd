
import numpy as np


# other --------------------------------------

# ====================================================================
def get_p_init(name):

    if name == "x^2":
        return( np.array([-3.0]) )
    elif name == "x^4":
        return( np.array([-3.0]) )
    elif name == "ellipse":
        return( np.array([4.0, 1.0]) )
    elif name == "2Dshell":
        return( np.array([4.0, 1.0]) )
    elif name == "1Dsigwell":
        return( np.array([-3.0]) )
    elif name == "Beale":
        return( np.array([4, 3]) )
#        return( np.array([1.5, 1.5]) )  #skip
#        return( np.array([-1, 4]) )
#        return( np.array([-2, -4]) )
    else:
        print("ERROR: no match in get_p_init, name = ", name)
        return(None)



# ====================================================================
def get_p_target(name):

    if name == "x^2":
        return( np.array([0.0]) )
    elif name == "x^4":
        return( np.array([0.0]) )
    elif name == "ellipse":
        return( np.array([0.0, 0.0]) )
    elif name == "2Dshell":
        return( np.array([-3.0, 0.0]) )  #the "-3" here isn't exact
    elif name == "1Dsigwell":
        return( np.array([0.0]) )
    elif name == "Beale":
        return( np.array([3, 0.5]) )
    else:
        print("ERROR: no match in get_p_init, name = ", name)
        return(None)


# ====================================================================
def sigmoid(z):
    return( 1/(1 + np.exp(-z)) )

# ====================================================================
def get_cost_names():
    return( ["x^2", "x^4", "ellipse", "1Dsigwell", "2Dshell", "Beale"] )

# ====================================================================
def get_cost(nameCF, p):
    if nameCF == "x^2":
        return getCost_x2(p)
    elif nameCF == "x^4":
        return getCost_x4(p)
    elif nameCF == "ellipse":
        return getCost_ellipse(p)
    elif nameCF == "1Dsigwell":
        return getCost_1Dsigwell(p)
    elif nameCF == "2Dshell":
        return getCost_2Dshell(p)
    elif nameCF == "Beale":
        return getCost_Beale(p)
    else:
        print("No match in get_cost()")

# ====================================================================
def get_grad(nameCF, p):
    if nameCF == "x^2":
        return getGrad_x2(p)
    elif nameCF == "x^4":
        return getGrad_x4(p)
    elif nameCF == "ellipse":
        return getGrad_ellipse(p)
    elif nameCF == "1Dsigwell":
        return getGrad_1Dsigwell(p)
    elif nameCF == "2Dshell":
        return getGrad_2Dshell(p)
    elif nameCF == "Beale":
        return getGrad_Beale(p)
    else:
        print("No match in get_grad()")
        
        
        
# x^2 ---------------------------------------------------------------
def getCost_x2(p):
    return(np.dot(p,p))   

def getGrad_x2(p):
    return( 2*p )   



# x^4 ---------------------------------------------------------------
def getCost_x4(p):
    return(np.dot(p,p) * np.dot(p,p))   

def getGrad_x4(p):
    return( 4 * p * np.dot(p,p) )



# ellipse -----------------------------------------------------------
def getParams_ellipse():
    return( {"a":4, "b":1} )

def getCost_ellipse(p):
    x = p[0]
    y = p[1]
    return getCost_ellipse_xy(x,y)

def getCost_ellipse_xy(x,y):
    dict_ = getParams_ellipse()
    a  = dict_["a"]
    b  = dict_["b"]
    return (x/a)**2 + (y/b)**2

def getGrad_ellipse(p):
    x = p[0]
    y = p[1]
    return getGrad_ellipse_xy(x,y)

def getGrad_ellipse_xy(x,y):
    dict_ = getParams_ellipse()
    a  = dict_["a"]
    b  = dict_["b"]
    grad_ = np.array( [2*x/a**2, 2*y/b**2] )
    return(grad_)


# 1Dsigwell -------------------------------------------------------------

# TO DO: allow override of these params
def getParams_1Dsigwell():
    return( {"s":10.0, "a":2.0} )

def getCost_1Dsigwell_x(x):
    dict_ = getParams_1Dsigwell()
    s = dict_["s"]
    a = dict_["a"]
    sig1 = sigmoid( s*( -x - a ) )
    sig2 = sigmoid( s*( x - a ) )
    return( sig1 + sig2 )

def getGrad_1Dsigwell_x(x):
    dict_ = getParams_1Dsigwell()
    s = dict_["s"]
    a = dict_["a"]
    sig1 = sigmoid( s*( -x - a ) )
    sig2 = sigmoid( s*( x - a ) )
    grad_ = -s*sig1*(1 - sig1) + s*sig2*(1 - sig2)
    return( grad_ )

def getCost_1Dsigwell(p):
    return( getCost_1Dsigwell_x(p[0]) )

def getGrad_1Dsigwell(p):
    return( getGrad_1Dsigwell_x(p[0]) )



# 2Dshell -------------------------------------------------------------

def getParams_2Dshell():
    return( {"a":1, "b":1, "rad":3, "s":3, "d":0.2, "zeta":0.01} )

# d_ = plus/minus difference in radii for two components of this cost fcn
#def getCost_2Dshell_xy(x_,y_,a_,b_,rad_,s_,d_,zeta_):
def getCost_2Dshell_xy(x,y):
    dict_ = getParams_2Dshell()
    a   = dict_["a"]
    b   = dict_["b"]
    rad = dict_["rad"]  # radius
    s   = dict_["s"]    # scale
    d   = dict_["d"]    # difference between radii
    zeta = dict_["zeta"]  # coefficient of tilt

    r1 = rad + d
    r2 = rad - d
    # interior well
    f1 = s*( (x/a)**2 + (y/b)**2 - r1*r1 )
    # interior bump
    f2 = -s*( (x/a)**2 + (y/b)**2 - r2*r2 )
    return( sigmoid(f1) + sigmoid(f2) + zeta*x ) 

def getCost_2Dshell(p):
    x = p[0]
    y = p[1]
    return getCost_2Dshell_xy(x,y)

def getGrad_2Dshell(p):
    # point
    x = p[0]
    y = p[1]
    # params
    dict_ = getParams_2Dshell()
    a   = dict_["a"]
    b   = dict_["b"]
    rad = dict_["rad"]  # radius
    s   = dict_["s"]    # scale
    d   = dict_["d"]    # difference between radii
    zeta = dict_["zeta"]  # coefficient of tilt

    # two radii
    r1 = rad + d
    r2 = rad - d
    # the well
    f1 = s*( (x/a)**2 + (y/b)**2 - r1*r1 )
    # the bump
    f2 = -s*( (x/a)**2 + (y/b)**2 - r2*r2 )
    # sigmoid derivatives
    sig1_der = sigmoid(f1) * (1 - sigmoid(f1))
    sig2_der = sigmoid(f2) * (1 - sigmoid(f2))
    fac = 2 * s * (sig1_der - sig2_der)
    # gradient
    grad_ = np.array( [x*fac/(a**2) + zeta, y*fac/(b**2)] )
    return(grad_ )



# Beale -------------------------------------------------------------

# f = (1.5 - x +xy )^2 + (2.25 -x + xy^2)^2 + (2.625 -x + xy^3)^2
# f_x = 2*(-1 + y)*(1.5 - x +xy ) + 2*(-1 + y^2)*(2.25 -x + xy^2) + 2*(-1 + y^3)*(2.625 -x + xy^3)
# f_y = 2*( x )*( 1.5 - x +xy ) + 2*( 2xy )*( 2.25 -x + xy^2 ) + 2*( 3xy^2 )*( 2.625 -x + xy^3 )
# global min at [3,0.5]

def getCost_Beale_xy(x,y):
    f = ( 1.5 - x +x*y )**2 + ( 2.25 -x + x*y**2 )**2 + ( 2.625 -x + x*y**3 )**2
    return( f )

def getCost_Beale(p):
    x = p[0]
    y = p[1]
    return getCost_Beale_xy(x,y)

def getGrad_Beale(p):
    # point
    x = p[0]
    y = p[1]
    f_x = 2*(-1 + y)*(1.5 - x + x*y ) + 2*(-1 + y*y)*(2.25 -x + x*y**2) + 2*(-1 + y**3)*(2.625 -x + x*y**3)
    f_y = 2*x*( 1.5 - x +x*y ) + 2*( 2*x*y )*( 2.25 -x + x*y**2 ) + 2*( 3*x*y**2 )*( 2.625 -x + x*y**3 )
    return( np.array([f_x,f_y]) )





import numpy as np
import matplotlib.pyplot as plt
import costgrad_vec as cg
import params as pm


# ====================================================================
def get_max_index(dict_):
    idx = -1
    for i in range(len(dict_["sumd_x"]),1,-1):
        if ~np.isnan(dict_["sumd_x"][i-1]) and ~np.isnan(dict_["sumd_y"][i-1]):
#            print("get_max_index: ", i-2, dict_["sumd_x"][i-2], dict_["sumd_y"][i-2] )
#            print("get_max_index: ", i-1, dict_["sumd_x"][i-1], dict_["sumd_y"][i-1] )
#            print("get_max_index: ", i, dict_["sumd_x"][i], dict_["sumd_y"][i] )
            idx = i-1
            break
    return(idx)


# ====================================================================
def do_Plots_final(dict_, rhomin, rhomax, sumd_line_freq):
    nrows = 7
    ncols = 3
    plt.figure(1, figsize=(14, 4*nrows))

    idx = 1
    # cost ---------------------------------------
    plt.subplot(nrows, ncols, idx)
    plt.plot( dict_["cost"] )
    plt.title("cost")

    # log(cost) --------------------------------
    idx += 1
    plt.subplot(nrows, ncols, idx)
    plt.plot( np.log10(dict_["cost"]) )
    plt.title("log_10(cost)")

    if "alpha" in dict_.keys():
        # alpha ---------------------------------------
        idx += 1
        plt.subplot(nrows, ncols, idx)
        plt.plot( dict_["alpha"] )
        plt.title("alpha")

        # log(alpha) --------------------------------
        idx += 1
        plt.subplot(nrows, ncols, idx)
        plt.plot( np.log10(dict_["alpha"]) )
        plt.title("log_10(alpha)")

    if "rho" in dict_.keys():
        # rho ---------------------------------------
        idx += 1
        plt.subplot(nrows, ncols, idx)
        plt.plot( dict_["rho"] )
        plt.title("rho")
        plt.axhline(y=rhomin, color='red', linewidth=0.5)
        plt.axhline(y=rhomax, color='red', linewidth=0.5)

        # log(rho) --------------------------------
        idx += 1
        plt.subplot(nrows, ncols, idx)
        plt.plot( np.log10(dict_["rho"]) )
        plt.title("log_10(rho)")
        plt.axhline(y=np.log10(rhomin), color='red', linewidth=0.5)
        plt.axhline(y=np.log10(rhomax), color='red', linewidth=0.5)

    if "dotp" in dict_.keys():
        # dotp ---------------------------------------
        idx += 1
        plt.subplot(nrows, ncols, idx)
        plt.plot( dict_["dotp"] )
        plt.title("dotp")
        plt.ylim(-1.3, 1.3)
        plt.axhline(y=1, color='red', linewidth=0.5)
        plt.axhline(y= -1, color='red', linewidth=0.5)

    if ("sumd_x" in dict_.keys()) and ("sumd_y" in dict_.keys()):
        # sumd ---------------------------------------
        idx += 1
        plt.subplot(nrows, ncols, idx)
        # blue line
        plt.plot( dict_["sumd_x"], dict_["sumd_y"] )
        # blue dots
        plt.plot( dict_["sumd_x"], dict_["sumd_y"], 'bo', markersize=3)
        
        # ideal line (dashed)
#        mx = dict_["sumd_x"][-1]
        ii = get_max_index(dict_)
        print("max index = ", ii)
        mx = dict_["sumd_x"][ii]
        print("final arclength (sumd_x) = ", mx)
        print("final distance (sumd_y) = ", dict_["sumd_y"][ii])
        
        plt.plot( [0,mx], [0,mx], 'r--', linewidth=0.5)
        # vertical lines
        plt.title("sumd")
        sumd_step = sumd_line_freq
        n = len(dict_["sumd_x"]) // sumd_step
        for i in range(n):
            ix = (i+1)*sumd_step
            if ix < len(dict_["sumd_x"]):
                v = dict_["sumd_x"][ix]
                plt.axvline(x=v, color='red', linewidth=0.5)

    if "incr" in dict_.keys():
        # incr ---------------------------------------
        idx += 1
        plt.subplot(nrows, ncols, idx)
        plt.plot( dict_["incr"] )
        plt.title("incr")

    if "decr" in dict_.keys():
        # decr ---------------------------------------
        idx += 1
        plt.subplot(nrows, ncols, idx)
        plt.plot( dict_["decr"] )
        plt.title("decr")

    if "p" in dict_.keys():
        # p ---------------------------------------
        idx += 1
        plt.subplot(nrows, ncols, idx)
        plt.plot( dict_["p"] )
        plt.title("p")
        
    if "rho_error" in dict_.keys():
        # rho_error ---------------------------------------
        idx += 1
        plt.subplot(nrows, ncols, idx)
        plt.plot( dict_["rho_error"] )
        plt.title("rho_error")
        plt.axhline(y= 1, color='red', linewidth=0.5)
        plt.axhline(y= 0, color='red', linewidth=0.5)
        plt.axhline(y= -1, color='red', linewidth=0.5)

    if "fac" in dict_.keys():
        # decr ---------------------------------------
        idx += 1
        plt.subplot(nrows, ncols, idx)
        plt.plot( dict_["fac"] )
        plt.title("fac")

    if "dist" in dict_.keys():
        # dist ---------------------------------------
        idx += 1
        plt.subplot(nrows, ncols, idx)
        plt.plot( dict_["dist"] )
        plt.title("distance to target")

        # log(dist) --------------------------------
        idx += 1
        plt.subplot(nrows, ncols, idx)
        plt.plot( np.log10(dict_["dist"]) )
        plt.title("log_10(distance to target)")

    plt.show()    


# ====================================================================
def contour_ellipsoid():
    x, y = np.meshgrid(np.arange(-5.0, 5.0, 0.2), np.arange(-5.0, 5.0, 0.2))
    fig, ax = plt.subplots(figsize=(10, 6))
    z = cg.getCost_ellipse_xy(x,y)
    
    # contours
    # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.contour.html
    
    cax = ax.contour(x, y, z, levels=np.linspace(0, 10, 10) )
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim((-5.0, 5.0))
    ax.set_ylim((-5.0, 5.0))
    
    return fig,ax


# ====================================================================
def do_plot_ellipse(px, py):
    fig4,ax4 = contour_ellipsoid()
    ax4.axhline(y=0, color='k', ls='--', lw=0.5)
    ax4.axvline(x=0, color='k', ls='--', lw=0.5)
    ax4.scatter( px, py, color='r', lw=0.6)
#    ax4.plot( px, py, color='r' )
    ax4.plot( px, py, color='r', markersize=3 )
    plt.show()


# ====================================================================
def do_plot_2Dshell(px,py):
    xlist = np.linspace(-5.0, 5.0, 100)
    ylist = np.linspace(-5.0, 5.0, 100)
    X, Y = np.meshgrid(xlist, ylist)
    Z = cg.getCost_2Dshell_xy(X,Y)

    plt.figure( figsize=(7,5) )
    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)
    
    size_title = 30
    size_label = 25
    plt.title('2Dshell', fontsize= size_title)
    plt.xlabel(r'$\theta_1$', fontsize= size_label)
    plt.ylabel(r'$\theta_2$', fontsize= size_label)

    plt.scatter( px, py, color='r', lw=0.9, s = 10)  #s =size of dots

    # start
    plt.plot(4,1, 'w*', markersize=20)
    # end
    plt.plot(-3,0, 'w*', markersize=20)

    plt.show()


# ====================================================================
def do_plot_special(name, d_results):
    if name == "ellipse" or name == "2Dshell":
        # populate px, py
        px=[]
        py=[]
        for c in d_results["p"]:
            px.append( c[0] )
            py.append( c[1] )

        if name == "ellipse":
            do_plot_ellipse(px, py)   
        elif name == "2Dshell":
            do_plot_2Dshell(px, py)

            

# ====================================================================
def do_plot_2Dshell_special(name, d_res, title, filename, ymin_left, ymax_left):
    if name != "2Dshell":
        return
    
    px=[]
    py=[]
    for c in d_res["p"]:
        px.append( c[0] )
        py.append( c[1] )

    size_title = 30
    size_label = 25
    
    nrows = 1
    ncols = 2
    plt.figure(1, figsize=(7*ncols, 5*nrows))

    # left figure (log(rho)) -----------------------
    idx = 1
    plt.subplot(nrows, ncols, idx)
    plt.plot( np.log10(d_res["rho"]), label="NeogradM")
    plt.title(r'$\log_{10} \rho$', fontsize= size_title)
#    plt.legend(loc = 'lower left')
    plt.xlabel(r'iterations', fontsize= size_label)
    plt.axhline(y=np.log10(pm.g_rhomin), color='red', linewidth=0.5)
    plt.axhline(y=np.log10(pm.g_rhomax), color='red', linewidth=0.5)

    if ymin_left is not None and ymax_left is not None:
        plt.ylim(ymin_left, ymax_left)
    

    # right figure (visual) -----------------------
    idx += 1
    plt.subplot(nrows, ncols, idx)
    
    xlist = np.linspace(-5.0, 5.0, 100)
    ylist = np.linspace(-5.0, 5.0, 100)
    X, Y = np.meshgrid(xlist, ylist)
    Z = cg.getCost_2Dshell_xy(X,Y)

    cp = plt.contourf(X, Y, Z)

    plt.colorbar(cp)
    plt.title(title, fontsize= size_title)
    plt.xlabel(r'$\theta_1$', fontsize= size_label)
    plt.ylabel(r'$\theta_2$', fontsize= size_label)
    
    # start
    plt.plot(4,1, 'w*', markersize=20)
    # end
    plt.plot(-3,0, 'w*', markersize=20)
    
    plt.scatter( px, py, color='r', lw=0.9, s = 10)  #s =size of dots
    
    if filename is not None:
        plt.savefig( filename + ".png")
    else:
        plt.show()

        
# ====================================================================
def get_dotp_sumd_plot( dict_, title_1, title_2, filename, vert_space, scale_ ):
    nrows = 1
    ncols = 2
    plt.figure(1, figsize=(6*ncols, 5*nrows))

    size_title = 25
    size_label = 20

    # dotp --------------------------------
    idx = 1
    plt.subplot(nrows, ncols, idx)
    plt.plot( dict_["dotp"] )
    plt.title( title_1, fontsize= size_title)
    plt.ylim(-1.3, 1.3)
    plt.axhline(y=1, color='red', linewidth=0.5)
    plt.axhline(y= -1, color='red', linewidth=0.5)
    plt.xlabel(r'iterations', fontsize= size_label)

    # sumd --------------------------------
    idx += 1    
    plt.subplot(nrows, ncols, idx)
    
    # blue line
    plt.plot( dict_["sumd_x"], dict_["sumd_y"] )
    # blue dots
    plt.plot( dict_["sumd_x"], dict_["sumd_y"], 'bo', markersize=3)
    
    # ideal line (dashed)
    if scale_ is None:
        mx = dict_["sumd_x"][-1]
    else:
        mx = scale_  #use this to have a comparable scale for both
    plt.plot( [0,mx], [0,mx], 'r--', linewidth=0.5)
    
    # vertical lines
    plt.title(title_2, fontsize= size_title)
    plt.xlabel(r'arclength', fontsize= size_label)
    plt.ylabel(r'distance', fontsize= size_label)
    sumd_step = vert_space
    n = len(dict_["sumd_x"]) // sumd_step
    for i in range(n):
        ix = (i+1)*sumd_step
        if ix < len(dict_["sumd_x"]):
            v = dict_["sumd_x"][ix]
            plt.axvline(x=v, color='red', linewidth=0.5)
            
    # save/show plot ------------------------
    if filename is not None:
        plt.savefig( filename )
    else:
        plt.show()



# ====================================================================
# Same as "get_dotp_sumd_plot", except it now includes different colored vertical lines every "vert_space_major" iterations
def get_dotp_sumd_plot_2( dict_, title_1, title_2, filename, vert_space_minor, vert_space_major, scale_ ):
    nrows = 1
    ncols = 2
    plt.figure(1, figsize=(6*ncols, 5*nrows))

    size_title = 25
    size_label = 20

    # dotp --------------------------------
    idx = 1
    plt.subplot(nrows, ncols, idx)
    plt.plot( dict_["dotp"] )
    plt.title( title_1, fontsize= size_title)
    plt.ylim(-1.3, 1.3)
    plt.axhline(y=1, color='red', linewidth=0.5)
    plt.axhline(y= -1, color='red', linewidth=0.5)
    plt.xlabel(r'iterations', fontsize= size_label)

    # sumd --------------------------------
    idx += 1    
    plt.subplot(nrows, ncols, idx)
    
    # blue line
    plt.plot( dict_["sumd_x"], dict_["sumd_y"] )
    # blue dots
    plt.plot( dict_["sumd_x"], dict_["sumd_y"], 'bo', markersize=3)
    
    # ideal line (dashed)
    if scale_ is None:
        mx = dict_["sumd_x"][-1]
    else:
        mx = scale_  #use this to have a comparable scale for both
    plt.plot( [0,mx], [0,mx], 'r--', linewidth=0.5)
    
    # vertical lines
    plt.title(title_2, fontsize= size_title)
    plt.xlabel(r'arclength', fontsize= size_label)
    plt.ylabel(r'distance', fontsize= size_label)
    sumd_step = vert_space_minor
    n = len(dict_["sumd_x"]) // sumd_step
    for i in range(n):
        ix = (i+1)*sumd_step
        if ix < len(dict_["sumd_x"]):
            if ix%vert_space_major==0 and ix != 0:
                v = dict_["sumd_x"][ix]
                plt.axvline(x=v, color='red', linewidth=0.5)
            else:
                v = dict_["sumd_x"][ix]
                plt.axvline(x=v, color='purple', linewidth=0.5)
            
    # save/show plot ------------------------
    if filename is not None:
        plt.savefig( filename )
    else:
        plt.show()
        
        

import numpy as np

class Global_Var:
    dvec_min=np.empty((0,))
    ref_point=np.empty((0,))
    obj_val_min=float("inf")
    best_hv=0
    test=0

    # Gaussian Processes related hyper-parameters
    max_cholesky_size = float("inf")  # Always use Cholesky
    size_Lanczos = 256
    chol_jitter = 1e-3    

    debug = False
    sim_cost = -1 # float(-1.) # seconds (-1 if real cost)
    average_cost = 0.2 # average cost of an evaluation, if real cost
    max_time = 20*60 #sim_cost * 80 # seconds, maximum time if budget in seconds

    if (sim_cost == (-1)):
        budget = int(max_time / average_cost); # in cycles, including initial DoE
    else:
        budget = int(max_time / sim_cost); 

    n_init = 296 # (int) (min((0.2*budget)*batch_size, 128));
    n_cycle = budget # (int) (0.8*budget);

    max_cholesky_size = 420 #float("inf")  # Always use Cholesky
    chol_jitter = 1e-3
    
    threshold = 500
    
    if debug:
        budget = 24
        n_init = 32
        n_cycle = budget

        # Budget allocated to the surrogate model fitting
        # (using GPytorch Models and custom fit function)
        threshold = 38
        large_fit = 50
        SKA_fit = 10
        small_fit = 10
        medium_fit = 20
        
        # Parameters of the acquisition strategies
        af_nrestarts = 3
        af_nsamples = 60
        af_options = {}
        af_options['maxfun']=200
        af_options['disp']=False
        af_options['maxiter']=20
        af_options['method']='L-BFGS-B'
    else :
        # Budget allocated to the surrogate model fitting
        # (using GPytorch Models and custom fit function)
        
        large_fit = 500
        SKA_fit = 50
        small_fit = 50
        medium_fit = 200
        
        # Parameters of the acquisition strategies
        af_nrestarts = 10
        af_nsamples = 512
        af_options = {}
        af_options['maxfun']=500
        af_options['disp']=False
        af_options['maxiter']=100
        af_options['method']='L-BFGS-B'
        

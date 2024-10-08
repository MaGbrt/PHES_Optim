import numpy as np
from Problems.Problem import Problem
from Problems.Single_Objective import Single_Objective
import matplotlib.pyplot as plt
from matplotlib import cm

class Alpine02(Single_Objective):
    """Class for the single-objective Alpine02 problem.

    :param dim: number of decision variable
    :type dim: positive int, >1
    """

    #-------------__init__-------------#    
    def __init__(self, n_dvar):
        assert n_dvar>=1
        Single_Objective.__init__(self, n_dvar)

        
    #-------------__del__-------------#
    def __del__(self):
        Problem.__del__(self)

            #-------------__str__-------------#
    def __str__(self):
        return "Alpine02 problem "+str(self.n_dvar)+" decision variables "+str(self.n_obj)+" objective"

    #-------------evaluate-------------#
    def perform_real_evaluation(self, c):
        """Objective function.

        :param candidates: candidate decision vectors
        :type candidates: np.ndarray
        :return: objective values
        :rtype: np.ndarray
        """
        candidates = np.copy(c)

        assert self.is_feasible(candidates)
        
        if candidates.ndim==1:
            candidates = np.array([candidates])

        candidates = (candidates + 100)/20 # Scale into [0, 10]^D, default for Alpine02, instead of [-100, 100] for CEC2015
        obj_vals = np.prod(np.sqrt(candidates)*np.sin(candidates), axis=1)

        return obj_vals

    #-------------get_bounds-------------#
    def get_bounds(self):
        """Returns search space bounds.

        :returns: search space bounds
        :rtype: np.ndarray
        """
        b=np.ones((2,self.n_dvar))
        b[0,:]*=-100
        b[1,:]*=100

        return b

    #-------------is_feasible-------------#
    def is_feasible(self, candidates):
        """Check feasibility of candidates.

        :param candidates: candidate decision vectors
        :type candidates: np.ndarray
        :returns: boolean indicating whether candidates are feasible
        :rtype: bool
        """

        res=False
        if Problem.is_feasible(self, candidates)==True:
            lower_bounds=self.get_bounds()[0,:]
            upper_bounds=self.get_bounds()[1,:]
            res=(lower_bounds<=candidates).all() and (candidates<=upper_bounds).all()
        return res

    #-------------plot-------------#
    def plot(self):
        """Plot the 1D or 2-D Alpine02 objective function."""
        
        if self.n_dvar==1:
            x = np.linspace(self.get_bounds()[0], self.get_bounds()[1], 100)
            y = self.perform_real_evaluation(x)

            plt.plot(x, y, 'r')
            plt.title("Alpine02-1D")
            plt.show()
            
        elif self.n_dvar==2:
            fig = plt.figure()

            lower_bounds = self.get_bounds()[0,:]
            upper_bounds = self.get_bounds()[1,:]

            x = np.linspace(lower_bounds[0], upper_bounds[0], 100)
            y = np.linspace(lower_bounds[1], upper_bounds[1], 100)
            z = self.perform_real_evaluation( np.array(np.meshgrid(x, y)).T.reshape(-1,2) ).reshape(x.size, y.size)
            x, y = np.meshgrid(x, y)
            
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet, antialiased=False)
            plt.title("Alpine02-2D")
            plt.show()

        else:
            print("[Alpine02.py] Impossible to plot Alpine02 with n_dvar="+str(self.n_dvar))

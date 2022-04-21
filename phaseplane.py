import numpy as np
import matplotlib.pyplot as plt
import sympy as sy



class Problem(object):
    def __init__(self, model, title, equation, symatrix):
        self.model = model
        self.title = title
        self.equation = equation
        if symatrix != None:
            self.symatrix = symatrix

def phase_plane_plot(problem, range_x = (-1,1), range_y = None,
                     num_grid_points = 1000, show = False, title=None):
    '''
    Simple implementation of the phase plane plot in matplotlib.

    Input:
    -----
      *model* : function
        function that takes numpy.array as input with two elements
        representing two state variables
      *range_x* = (-1, 1) : tuple
        range of x axis
      *range_y* = None : tuple
        range of y axis; if None, the same range as *range_x*
      *num_grid_points* = 50 : int
        number of samples on grid
      *show* = False : bool
        if True it shows matplotlib plot
    '''
    print("\n\n",problem.title, "\n")
    SympyProblem(problem.symatrix, problem.title)

    if range_y is None:
        range_y = range_x
    x_ = np.linspace(range_x[0], range_x[1], num_grid_points)
    y_ = np.linspace(range_y[0], range_y[1], num_grid_points)

    grid = np.meshgrid(x_, y_)

    eq_points = []
    dfmat = np.zeros((num_grid_points, num_grid_points, 2))
    for nx in range(num_grid_points):
        for ny in range(num_grid_points):
            df = problem.model([grid[0][nx,ny], grid[1][nx,ny]])
            dfmat[nx, ny, 0] = df[0]
            dfmat[nx, ny, 1] = df[1]

    plt.figure(problem.title)
    plt.title(problem.equation)
    # plt.quiver(grid[0], grid[1], dfmat[:, :, 0], dfmat[:, :, 1], headwidth=0.2, width=0.01 )
    plt.streamplot(grid[0], grid[1], dfmat[:, :, 0], dfmat[:, :, 1])
    plt.contour(grid[0], grid[1], dfmat[:, :, 0], [0], colors = 'r')
    plt.contour(grid[0], grid[1], dfmat[:, :, 1], [0], colors = 'g')

      # Outputs folder
    import os
    result_path = os.path.join(os.path.dirname(__file__), "results")
    if os.path.exists(result_path) is not True:
        os.mkdir(result_path)
    plt.savefig(result_path+"/"+problem.title+".jpg")
    if show:
        plt.show()

def PlotProblems(show_plot = False):
    x, y = sy.symbols('x y')

    p1 = Problem(lambda x: [-2*x[0]*(x[0]-1)*(2*x[0]-1), -2*x[0]], "problem1",\
            r'$d/dt [x, y] = [-2x(x-1)(2x-1), -2y]$', \
            sy.Matrix([ -2*x*(x-1)*(2*x-1), -2*y ]))
    phase_plane_plot(problem=p1, range_x = (-5, 5), show = show_plot)

    p2 = Problem(lambda x: [x[0]*(4-2*x[0]-x[1]), x[1]*(3-x[0] -x[1])], "problem2", \
            r'$d/dt [x, y] = [x(4-2x-y), y(3-x-y)]$', \
            sy.Matrix([ x*(4-2*x-y), y*(3-x-y) ]))
    phase_plane_plot(problem=p2, range_x = (-5, 5), show = show_plot)

    p3 = Problem(lambda x: [x[0]*(60-4*x[0]-3*x[1]), x[1]*(42-3*x[0] -2*x[1])], "problem3", \
            r'$d/dt [x, y] = [x(60-4x-3y), y(42-3x-2y)]$', \
            sy.Matrix([ x*(60-4*x-3*y), y*(42-3*x-2*y) ]))
    phase_plane_plot(problem=p3, range_x = (-600, 600), show = show_plot)

    p4 = Problem(lambda x: [2*x[0] - x[0]*x[1], 2*x[0]*x[1] - x[1]**2 - 2*x[1]], "problem4", \
            r'$d/dt [x, y] = [2x - xy, 2xy - y^2 - 2y]$', \
            sy.Matrix([ 2*x - x*y, 2*x*y - y**2 - 2*y ]))
    phase_plane_plot(problem=p4, range_x = (-4, 4), show = show_plot)

    p5 = Problem(lambda x: [-x[0]*(2- x[0]**2 - x[1]**2), -x[1]*(1 + x[0]**2 + x[1]**2 - 3*x[0])], "problem5", \
            r'$d/dt [x, y] = [-x(2 - x^2 - y^2), -y(1 + x^2 + y^2 - 3x)]$', \
            sy.Matrix([ -x*(2 - x**2 - y**2), -y*(1 + x**2 + y**2 - 3*x) ]))
    phase_plane_plot(problem=p5, range_x = (-2, 2), show = show_plot)

    p6 = Problem(lambda x: [np.sin(x[0])*np.cos(x[1]),np.sin(x[1])*np.cos(x[0])], "problem6", \
            r'$d/dt [x, y] = [\sin(x) \cos(y), \sin(y) \cos(x)]$',
            sy.Matrix([ sy.sin(x)* sy.cos(y), sy.sin(y)* sy.cos(x) ]))
    phase_plane_plot(problem=p6, range_x = (-6, 6), show = show_plot)


def SympyProblem(F: sy.Matrix, title):

    from contextlib import redirect_stdout
    with open('results/' + title+'.txt', 'w') as f:
        with redirect_stdout(f):

            x, y = sy.symbols('x y')
            X = sy.Matrix([ x,\
                            y ])
            equi_points = sy.solve(F, dict=True)
            DF = sy.simplify(F.jacobian(X))

            print("Let us define the system, X' = F(X) ")
            sy.pprint(X)

            print("= ")
            sy.pprint(F)

            print("With the equilibrium points")
            sy.pprint(equi_points)

            print("\n The F Jacobian is ")
            print("")
            sy.pprint(DF)

            print("\n The classified equilibrium points are ")
            for i in range( len(equi_points) ):
                point = equi_points[i]
                DF_point = DF.subs([ (x,point[x]), (y, point[y]) ])

                print("Eq point ", i, "is", '->', point[x], point[y], "with DF matrix: " )
                sy.pprint(DF_point)

                DF_point_np = np.array(DF_point.tolist()).astype(np.cdouble)

                if np.linalg.det( DF_point_np) > 0 and np.trace(DF_point_np) < 0:
                    print("Point is sink")
                if np.linalg.det(DF_point_np) > 0 and np.trace(DF_point_np) > 0:
                    print("Point is source")
                if np.linalg.det(DF_point_np) < 0:
                    print("Point is saddle")



if __name__ == "__main__":

    show_plot_end = False
    PlotProblems()


    if show_plot_end is True:
        plt.show()

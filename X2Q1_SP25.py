#utilized ChatGPT
# region imports
import numpy as np
from scipy.integrate import quad, solve_ivp
import matplotlib.pyplot as plt
# endregion

# region function definitions
def S(x):
    """
    Computes the integral S(x) = ∫[0 to x] sin(t^2) dt using scipy.integrate.quad.
    This integral is part of the exact solution's definition.
    """
    result = quad(lambda t: np.sin(t**2), 0, x)  # integrate sin(t^2) from 0 to x
    return result[0]  # the integral value is in result[0]

def Exact(x):
    """
    Returns the exact solution y(x) = 1 / [2.5 - S(x)] + 0.01*x^2,
    where S(x) is the Fresnel-like integral of sin(t^2) from 0 to x.
    """
    return 1 / (2.5 - S(x)) + 0.01 * (x**2)  # formula for the exact solution

def ODE_System(x, y):
    """
    Defines the ODE: y' = (y - 0.01*x^2)^2 * sin(x^2) + 0.02*x.
    This is passed to solve_ivp to compute the numerical solution.
    """
    Y = y[0]  # we have a single state variable y
    Ydot = (Y - 0.01 * (x**2))**2 * np.sin(x**2) + 0.02 * x  # derivative at x
    return [Ydot]

def Plot_Result(*args):
    """
    Plots the exact solution as a solid line and the numerical solution
    as upward-facing triangles, then formats the axes to match the problem's
    requirements: x from 0 to 6, y from 0 to 1.
    """
    xRange_Num, y_Num, xRange_Xct, y_Xct = args  # unpack the plot arrays
    plt.plot(xRange_Xct, y_Xct, 'b-', label='Exact')   # solid blue line for exact
    plt.plot(xRange_Num, y_Num, 'r^', label='Numerical')  # red triangles for numerical
    plt.xlabel("x")  # label x-axis
    plt.ylabel("y")  # label y-axis
    plt.title("IVP: y'=(y-0.01x^2)^2 sin(x^2)+0.02x, y(0)=0.4")  # set the title
    plt.legend()  # show the legend
    plt.xlim(0, 6)  # force x-axis to range from 0 to 6
    plt.xticks(np.arange(0, 6.1, 1.0))  # x ticks at 0.0, 1.0, 2.0, ..., 6.0
    plt.ylim(0, 1)  # force y-axis to range from 0 to 1
    plt.yticks(np.arange(0, 1.1, 0.2))  # y ticks at 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    plt.show()  # display the plot
# endregion

def main():
    """
    Solves the IVP y'=(y-0.01x^2)^2 sin(x^2)+0.02x, y(0)=0.4 using solve_ivp
    with a step size of 0.2 over x from 0 to 5. It also computes the exact
    solution via the Fresnel-like integral S(x) and plots both for comparison.
    """
    xRange = np.arange(0, 5.01, 0.2)  # create x array for numerical solution (0 to 5, step 0.2)
    xRange_xct = np.linspace(0, 5, 500)  # create a finer x array for the exact solution
    Y0 = [0.4]  # initial condition: y(0)=0.4
    sln = solve_ivp(ODE_System, [0, 5], Y0, t_eval=xRange)  # solve the ODE numerically
    xctSln = np.array([Exact(x) for x in xRange_xct])  # exact y-values for each point in xRange_xct
    Plot_Result(xRange, sln.y[0], xRange_xct, xctSln)  # compare numerical & exact solutions on one plot
    pass

# region function calls
if __name__ == "__main__":
    main()
# endregion
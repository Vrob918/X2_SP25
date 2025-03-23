#utilized ChatGPT
# region imports
from scipy.integrate import solve_ivp   # For numerically integrating ODEs
import math                             # For math.sin() and other math functions
import numpy as np                      # For array operations and linspace
import matplotlib.pyplot as plt         # For plotting
# endregion

# region class definitions
class circuit():
    def __init__(self, R=20, L=20, C=0.05, A=20, w=20, p=0):
        """
        Initializes an RLC circuit with default or user-provided parameters.

        :param R: Resistance in ohms (Ω)
        :param L: Inductance in henries (H)
        :param C: Capacitance in farads (F)
        :param A: Amplitude of voltage source (volts)
        :param w: Angular frequency ω of the source (rad/s)
        :param p: Phase φ of the source (radians)
        """
        self.R = R          # Store the resistance
        self.L = L          # Store the inductance
        self.C = C          # Store the capacitance
        self.A = A          # Store the amplitude of the driving voltage
        self.w = w          # Store the angular frequency of the driving voltage
        self.p = p          # Store the phase of the driving voltage

        # Attributes to store the simulation results after solving ODEs
        self.t = None       # Will hold the time array
        self.vC = None      # Will hold the capacitor voltage over time
        self.i1 = None      # Will hold i1(t) (inductor current in this setup)
        self.i2 = None      # Will hold i2(t) (capacitor current in this setup)

    def ode_system(self, t, X):
        """
        Defines the system of ODEs for the circuit.

        State variables:
          X[0] = iL(t)  -> inductor current
          X[1] = vC(t)  -> capacitor voltage

        :param t: Current time (float)
        :param X: Current state variables [iL, vC]
        :return: List of derivatives [diL/dt, dvC/dt]
        """
        iL = X[0]   # Extract inductor current from state vector
        vC = X[1]   # Extract capacitor voltage from state vector

        # Driving voltage: v_in(t) = A * sin(w * t + p)
        v_in = self.A * math.sin(self.w * t + self.p)

        # From KVL: v_in = L * diL/dt + vC
        # => diL/dt = (v_in - vC) / L
        diL_dt = (v_in - vC) / self.L

        # From KCL at the node: iL = iR + iC
        # iR = vC/R, iC = C * dvC/dt
        # => dvC/dt = (iL - iR) / C = [iL - (vC/R)] / C
        dvC_dt = (iL - (vC / self.R)) / self.C

        return [diL_dt, dvC_dt]  # Return the derivatives of iL and vC

    def simulate(self, t=10, pts=500):
        """
        Simulates the transient behavior of the RLC circuit from t=0 to t (default 10s).

        :param t: End time for simulation in seconds (default 10)
        :param pts: Number of points for the time array (default 500)
        """
        # Create a time array from 0 to t with pts points
        t_eval = np.linspace(0, t, pts)

        # Initial conditions: iL(0)=0, vC(0)=0
        X0 = [0.0, 0.0]

        # Solve the ODE system using solve_ivp
        sol = solve_ivp(self.ode_system, [0, t], X0, t_eval=t_eval)

        # Extract the time array from the solver
        self.t = sol.t

        # Extract the inductor current and capacitor voltage from the solution
        iL_array = sol.y[0]  # This corresponds to iL(t)
        vC_array = sol.y[1]  # This corresponds to vC(t)

        # i1(t) = inductor current (straight from X[0])
        self.i1 = iL_array

        # i2(t) = capacitor current = C * dvC/dt
        # We approximate dvC/dt via np.gradient
        dvC_dt = np.gradient(vC_array, self.t)
        iC_array = self.C * dvC_dt
        self.i2 = iC_array

        # Flip the capacitor voltage sign to match the reference diagram's polarity
        self.vC = -vC_array

    def doPlot(self):
        """
        Produces a 2D plot with two y-axes:
          - Left y-axis for i1(t) and i2(t) in Amperes
          - Right y-axis for vC(t) in Volts

        Uses major/minor gridlines, custom axis ranges/ticks, and
        distinct line styles for each data series.
        """
        # Create a figure and one Axes (ax1)
        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Plot i1(t) (inductor current) as a solid black line
        ax1.plot(self.t, self.i1, 'k-', label='i1(t) = iL(t)')

        # Plot i2(t) (capacitor current) as a dashed black line
        ax1.plot(self.t, self.i2, 'k--', label='i2(t) = iC(t)')

        # Label the x-axis, left y-axis, and figure title
        ax1.set_xlabel("t (s)")
        ax1.set_ylabel("i1, i2 (A)")
        ax1.set_title("RLC Circuit: i1(t), i2(t), and vC(t) vs. Time")

        # Set the x-axis limit from 0..10
        ax1.set_xlim(0, 10)
        # Major ticks at each integer 0..10, minor ticks every 0.5
        ax1.set_xticks(np.arange(0, 11, 1.0))
        ax1.set_xticks(np.arange(0, 10.1, 0.5), minor=True)

        # Left y-axis for currents: range -0.06..0.10
        ax1.set_ylim(-0.06, 0.10)
        # Major ticks every 0.02, minor ticks every 0.01
        ax1.set_yticks(np.arange(-0.06, 0.101, 0.02))
        ax1.set_yticks(np.arange(-0.06, 0.101, 0.01), minor=True)

        # Create a second y-axis (ax2) sharing the same x-axis
        ax2 = ax1.twinx()
        # Plot vC(t) as a dotted black line on ax2
        ax2.plot(self.t, self.vC, 'k:', label='vC(t)')
        ax2.set_ylabel("vC(t) (V)")

        # Right y-axis from -0.5..0.1
        ax2.set_ylim(-0.5, 0.1)
        ax2.set_yticks(np.arange(-0.5, 0.11, 0.1))
        ax2.set_yticks(np.arange(-0.5, 0.11, 0.05), minor=True)

        # Merge the legends from both axes so they appear in one box
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        # Add major and minor gridlines in light gray
        ax1.grid(which='major', color='lightgray', linewidth=1)
        ax1.grid(which='minor', color='lightgray', linestyle=':', linewidth=0.5)

        # Tighten up the layout so labels don’t overlap
        plt.tight_layout()

        # Display the figure
        plt.show()
# endregion

# region function definitions
def main():
    """
    Main function for running the RLC circuit simulation and plotting.
    Allows user to update circuit parameters and re-run if desired.
    """
    goAgain = True   # Boolean to control the simulation loop

    # Create a circuit object with default parameters:
    # R=10 Ω, L=20 H, C=0.05 F, A=20 V amplitude, w=20 rad/s, p=0 rad phase
    Circuit = circuit(R=10, L=20, C=0.05, A=20, w=20, p=0)

    while goAgain:
        try:
            # Prompt user for new parameter values
            R_in = float(input("Enter R (ohms) [default=10]: "))
            L_in = float(input("Enter L (H) [default=20]: "))
            C_in = float(input("Enter C (F) [default=0.05]: "))
            A_in = float(input("Enter amplitude A (volts) [default=20]: "))
            w_in = float(input("Enter angular frequency w (rad/s) [default=20]: "))
            p_in = float(input("Enter phase p (radians) [default=0]: "))

            # Update the circuit object’s parameters
            Circuit.R = R_in
            Circuit.L = L_in
            Circuit.C = C_in
            Circuit.A = A_in
            Circuit.w = w_in
            Circuit.p = p_in

        except ValueError:
            # If user inputs something invalid, retain old or default values
            print("Invalid input; using existing or default circuit values.")

        # Run the simulation for 10 seconds, storing 500 points
        Circuit.simulate(t=10, pts=500)
        # Plot the results
        Circuit.doPlot()

        # Ask if the user wants to repeat the simulation with new parameters
        again = input("Run another simulation? (y/n): ")
        if not again.lower().startswith('y'):
            goAgain = False  # If answer is not 'y', exit the loop

if __name__ == "__main__":
    main()
# endregion
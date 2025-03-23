#utilized HW6
#utilized ChatGPT
# region imports
import numpy as np
import math
from scipy.optimize import fsolve
import random as rnd
# endregion

# region class definitions
class UC():  # Define a class UC (Units Converter)
    def __init__(self):  # Define the constructor method for UC
        """
        This unit converter class is useful for the pipe network and perhaps other problems.
        The strategy is (number in current units)*(conversion factor)=(number desired units), for instance:
            1(ft)*(self.ft_to_m) = 1/3.28084 (m)
            1(in^2)*(self.in2_to_m2) = 1*(1/(12*3.28084))**2 (m^2)
        """
        pass

    # region class constants
    # we have not used these in class yet, but I think they are not too mysterious.
    ft_to_m = 1 / 3.28084
    ft2_to_m2 = ft_to_m ** 2
    ft3_to_m3 = ft_to_m ** 3
    ft3_to_L = ft3_to_m3 * 1000
    L_to_ft3 = 1 / ft3_to_L
    in_to_m = ft_to_m / 12
    m_to_in = 1 / in_to_m
    in2_to_m2 = in_to_m ** 2
    m2_to_in2 = 1 / in2_to_m2
    g_SI = 9.80665
    g_EN = 32.174
    gc_EN = 32.174
    gc_SI = 1.0
    lbf_to_kg = 1 / 2.20462
    lbf_to_N = lbf_to_kg * g_SI
    pa_to_psi = (1 / (lbf_to_N)) * in2_to_m2
    # endregion

    @classmethod  # Indicates a class method (no need for an instance to call it)
    def viscosityEnglishToSI(cls, mu, toSI=True):  # Define method to convert viscosity between English and SI
        cf = (1 / cls.ft2_to_m2) * (cls.lbf_to_kg) * cls.g_SI  # Calculate conversion factor
        return mu * cf if toSI else mu / cf  # Return converted viscosity based on toSI flag

    @classmethod
    def densityEnglishToSI(cls, rho, toSI=True):  # Define method to convert density between English and SI
        cf = cls.lbf_to_kg / cls.ft3_to_m3      # Calculate conversion factor
        return rho * cf if toSI else rho / cf   # Return converted density based on toSI flag

    @classmethod
    def head_to_pressure(cls, h, rho, SI=True):  # Convert fluid head (height) to pressure in consistent units
        if SI:  # If using SI units
            cf = rho * cls.g_SI / cls.gc_SI  # Calculate SI-based conversion
            return h * cf                   # Return pressure in Pa (or N/m^2)
        else:  # If using English units
            cf = rho * cls.g_EN / cls.gc_EN * (1 / 12) ** 2  # Calculate English-based conversion
            return h * cf                                   # Return pressure in lbf/ft^2 or similar

    @classmethod
    def m_to_psi(cls, h, rho):  # Convert meters of fluid column to psi
        return cls.head_to_pressure(h, rho) * cls.pa_to_psi  # Use head_to_pressure, then multiply by pa_to_psi

    @classmethod
    def psi_to_m(cls, p, rho):  # Convert psi to meters of fluid
        pa = p / cls.pa_to_psi  # First convert psi to Pa
        h = pa / (rho * cls.g_SI)  # Then convert Pa to head in meters
        return h                  # Return the height in meters


class Fluid():  # Define Fluid class to model fluid properties
    """
    Provides a fluid model with viscosity, density, and computed kinematic viscosity.
    """
    def __init__(self, mu=0.00089, rho=1000, SI=True):  # Constructor for Fluid
        # Set fluid properties; convert from English if SI=False
        self.mu = mu if SI else UC.viscosityEnglishToSI(mu)  # Dynamic viscosity
        self.rho = rho if SI else UC.densityEnglishToSI(rho)  # Density
        self.nu = self.mu / self.rho  # Kinematic viscosity: nu = mu / rho


class Node():  # Define Node class to represent a junction or node in the network
    """
    Represents a node in the pipe network, tracking connected pipes,
    external flow, net flow, and computed pressure.
    """
    def __init__(self, Name='a', Pipes=[], ExtFlow=0):  # Constructor for Node
        # Basic node properties
        self.name = Name      # Node name (string)
        self.pipes = Pipes    # List of connected pipes
        self.extFlow = ExtFlow  # External flow at node (+inflow, -outflow)
        self.QNet = 0         # Net flow (will be computed)
        self.P = 0            # Node pressure/head (will be computed)
        self.oCalculated = False  # Flag for tracking if node has been processed

    def getNetFlowRate(self):  # Calculate net flow into this node
        """
        Summarizes inflow minus outflow for the node by iterating over connected pipes.
        """
        Qtot = self.extFlow  # Start with external flow
        for p in self.pipes:  # For each connected pipe
            Qtot += p.getFlowIntoNode(self.name)  # Add or subtract based on flow direction
        self.QNet = Qtot  # Store final net flow
        return self.QNet  # Return net flow

    def setExtFlow(self, E, SI=True):  # Set external flow at this node
        """
        Assigns an external flow rate (in or out) to the node, converting
        to consistent units if necessary.
        """
        self.extFlow = E if SI else E * UC.ft3_to_L  # Convert from ft^3/s to L/s if needed


class Loop():  # Define Loop class to represent a closed circuit of pipes
    """
    Defines a closed loop in the network, composed of pipes arranged
    in order to allow head-loss computations around the circuit.
    """
    def __init__(self, Name='A', Pipes=[]):  # Constructor for Loop
        # Loop properties: name and the ordered list of pipes
        self.name = Name
        self.pipes = Pipes

    def getLoopHeadLoss(self):  # Compute cumulative head loss around this loop
        """
        Computes cumulative head loss following the specified pipe sequence.
        """
        deltaP = 0  # Initialize total head loss
        startNode = self.pipes[0].startNode  # Get the start node from the first pipe
        for p in self.pipes:  # For each pipe in the loop
            phl = p.getFlowHeadLoss(startNode)  # Get the signed head loss for each pipe
            deltaP += phl  # Add to total
            startNode = p.endNode if startNode != p.endNode else p.startNode  # Move to the other end
        return deltaP  # Return total loop head loss


class Pipe():  # Define Pipe class to represent an individual pipe segment
    """
    Represents an individual pipe with geometric properties, fluid data,
    and methods to calculate head loss and flow direction.
    """
    def __init__(self, Start='A', End='B', L=100, D=200, r=0.00025, fluid=Fluid(), SI=True):  # Constructor
        # Use alphabetical order for start/end node
        self.startNode = min(Start.lower(), End.lower())  # Lower letter as start
        self.endNode = max(Start.lower(), End.lower())    # Higher letter as end
        self.length = L if SI else UC.ft_to_m * L         # Pipe length (convert if needed)
        self.rough = r if SI else UC.ft_to_m * r          # Pipe roughness (convert if needed)
        self.fluid = fluid                                # Fluid object
        self.d = D / 1000.0 if SI else UC.in_to_m * D      # Pipe diameter in meters (convert if needed)
        self.relrough = self.rough / self.d               # Relative roughness
        self.A = math.pi / 4.0 * self.d ** 2              # Cross-sectional area
        self.Q = 10                                       # Initial guess for flow (L/s)
        self.vel = self.V()                               # Velocity
        self.reynolds = self.Re()                         # Reynolds number
        self.hl = 0                                       # Head loss

    def V(self):  # Compute average velocity
        """
        Calculates and stores the average velocity based on current flow rate Q.
        """
        self.vel = abs(self.Q) / (1000.0 * self.A)  # Convert Q from L/s to m^3/s, then divide by area
        return self.vel  # Return velocity

    def Re(self):  # Compute Reynolds number
        """
        Recomputes the Reynolds number using updated velocity, diameter, and fluid properties.
        """
        v = self.V()  # Ensure velocity is up to date
        self.reynolds = self.fluid.rho * v * self.d / self.fluid.mu  # Re = rho * v * d / mu
        return self.reynolds  # Return Reynolds number

    def FrictionFactor(self):  # Compute friction factor using laminar, turbulent, or transitional logic
        """
        Determines the Darcy friction factor based on Reynolds number and relative roughness.
        For flows in the transition regime, it inserts randomness to simulate uncertainty.
        """
        Re = self.Re()             # Get current Reynolds number
        rr = self.relrough         # Relative roughness

        def CB():  # Define Colebrook function for turbulent flow
            cb = lambda f: 1 / (f ** 0.5) + 2.0 * np.log10(rr / 3.7 + 2.51 / (Re * f ** 0.5))
            result = fsolve(cb, 0.01)  # Solve for friction factor
            return result[0]

        def lam():  # Define laminar friction factor formula
            return 64 / Re

        if Re >= 4000:  # Turbulent regime
            return CB()
        elif Re <= 2000:  # Laminar regime
            return lam()
        else:
            # Transition region: do a linear interpolation plus some random scatter
            CBff = CB()  # Turbulent friction factor
            Lamff = lam() # Laminar friction factor
            alpha = (Re - 2000) / (4000 - 2000)  # Fraction across transition
            mean = alpha * CBff + (1 - alpha) * Lamff  # Weighted average
            # Compute standard deviation for random variation
            sig_1 = (1 - (Re - 3000) / 1000) * 0.2 * mean
            sig_2 = (1 - (3000 - Re) / 1000) * 0.2 * mean
            sig = sig_1 if Re >= 3000 else sig_2
            return rnd.normalvariate(mean, sig)  # Return random deviate around mean

    def frictionHeadLoss(self):  # Calculate Darcy-Weisbach head loss
        """
        Uses the Darcy-Weisbach approach to calculate head loss along the pipe
        for the current flow conditions.
        """
        g = 9.81  # Acceleration due to gravity (m/s^2)
        ff = self.FrictionFactor()  # Get friction factor
        v = self.V()                # Ensure velocity is updated
        self.hl = ff * (self.length / self.d) * (v ** 2 / (2 * g))  # Darcy-Weisbach formula
        return self.hl  # Return head loss

    def getFlowHeadLoss(self, s):  # Return signed head loss given a start node
        """
        Returns a signed head loss (positive or negative) depending on traversal
        direction and flow direction.
        """
        nTraverse = 1 if s == self.startNode else -1  # +1 if traveling from startNode, -1 otherwise
        nFlow = 1 if self.Q >= 0 else -1              # +1 if Q is positive (start->end)
        return nTraverse * nFlow * self.frictionHeadLoss()  # Product of signs times frictional HL

    def Name(self):  # Return string name for the pipe
        """
        Returns a string label identifying this pipe by its node endpoints.
        """
        return self.startNode + '-' + self.endNode  # e.g., "a-b"

    def oContainsNode(self, node):  # Check if pipe connects to a specified node
        """
        Checks whether a given node name matches one of this pipe's endpoints.
        """
        return self.startNode == node or self.endNode == node

    def printPipeFlowRate(self, SI=True):  # Print flow rate in L/s or cfs
        # Print flow in either L/s (SI) or cfs (English)
        q_units = 'L/s' if SI else 'cfs'
        q = self.Q if SI else self.Q * UC.L_to_ft3
        print('The flow in segment {} is {:0.2f} ({}) and Re={:.1f}'.format(self.Name(), q, q_units, self.reynolds))

    def printPipeHeadLoss(self, SI=True):  # Print head loss in mm or inches
        # Calculate conversion factors for diameter, length, and head
        cfd = 1000 if SI else UC.m_to_in    # multiplier for diameter: to mm or in
        unitsd = 'mm' if SI else 'in'       # diameter units
        cfL = 1 if SI else 1 / UC.ft_to_m   # multiplier for length: to m or in
        unitsL = 'm' if SI else 'in'        # length units
        cfh = cfd                           # multiplier for head: same as diameter
        units_h = unitsd                    # head units: same as diameter units
        print("head loss in pipe {} (L={:.2f} {}, d={:.2f} {}) is {:.2f} {} of water".format(
            self.Name(), self.length * cfL, unitsL, self.d * cfd, unitsd, self.hl * cfh, units_h))

    def getFlowIntoNode(self, n):  # Return flow rate into a given node
        """
        Returns the flow into a specified node (positive or negative sign)
        depending on the node's relationship to this pipe's direction.
        """
        if n == self.startNode:  # If node is pipe's start, flow is leaving => negative
            return -self.Q
        return self.Q            # Otherwise, flow is entering => positive


class PipeNetwork():  # Define PipeNetwork class to manage entire system
    '''
    The pipe network is built from pipe, node, loop, and fluid objects.
    :param Pipes: a list of pipe objects
    :param Loops: a list of loop objects
    :param Nodes: a list of node objects
    :param fluid: a fluid object
    '''
    def __init__(self, Pipes=[], Loops=[], Nodes=[], fluid=Fluid()):  # Constructor
        """
        Manages multiple pipes, nodes, loops, and a fluid definition,
        supporting flow/pressure calculations across the network.
        """
        self.loops = Loops     # Store loops in the network
        self.nodes = Nodes     # Store nodes in the network
        self.Fluid = fluid     # Store fluid properties
        self.pipes = Pipes     # Store pipes in the network

    def findFlowRates(self):  # Main solver routine to find flow rates
        '''
        a method to analyze the pipe network and find the flow rates in each pipe
        given the constraints of: i) no net flow into a node and ii) no net pressure drops in the loops.
        :return: a list of flow rates in the pipes
        '''
        N = len(self.nodes) + len(self.loops)  # total number of equations: node continuity + loop equations
        Q0 = np.full(N, 10)                    # initial guess for all flow rates

        def fn(q):  # nested function for fsolve callback
            """
            This is used as a callback for fsolve.  The mass continuity equations at the nodes and the loop equations
            are functions of the flow rates in the pipes.  Hence, fsolve will search for the roots of these equations
            by varying the flow rates in each pipe.
            :param q: an array of flowrates in the pipes + 1 extra value b/c of node b
            :return: L an array containing flow rates at the nodes and  pressure losses for the loops
            """
            for n in self.nodes:     # Reset node pressure flags
                n.P = 0
                n.oCalculated = False
            for i in range(len(self.pipes)):  # Assign guessed flow rates to pipe objects
                self.pipes[i].Q = q[i]
            L = self.getNodeFlowRates()     # Evaluate node continuity
            L += self.getLoopHeadLosses()   # Evaluate loop constraints
            return L                        # Return combined residual

        FR = fsolve(fn, Q0)  # Solve the system for zero residual
        return FR            # Return final flow rate array

    def getNodeFlowRates(self):  # Gather net flows from each node
        """
        Assembles each node's net flow rate into a list for solver logic.
        """
        qNet = [n.getNetFlowRate() for n in self.nodes]  # list comprehension over all nodes
        return qNet

    def getLoopHeadLosses(self):  # Gather head losses from each loop
        """
        Assembles each loop's net head loss into a list, used by solver to enforce loop constraints.
        """
        lhl = [l.getLoopHeadLoss() for l in self.loops]  # list comprehension over all loops
        return lhl

    def getNodePressures(self, knownNodeP, knownNode):  # Compute node pressures from known reference
        '''
        Calculates the pressures at the nodes by traversing the loops.  For this to work,
        I must traverse the nodes in the proper order, so that the start node of each loop except
        for the first one has been calculated before traversing the loop.
        :return:
        '''
        for n in self.nodes:  # Reset all node pressures
            n.P = 0.0
            n.oCalculated = False
        for l in self.loops:  # Traverse each loop in turn
            startNode = l.pipes[0].startNode  # get the loop's first pipe's start node
            nd = self.getNode(startNode)      # get Node object
            CurrentP = nd.P                   # retrieve current pressure
            nd.oCalculated = True             # mark node as calculated
            for p in l.pipes:                # for each pipe in this loop
                phl = p.getFlowHeadLoss(startNode)  # get signed head loss
                CurrentP -= phl                    # subtract from current pressure
                startNode = p.endNode if startNode != p.endNode else p.startNode  # move to the next node
                nd = self.getNode(startNode)       # retrieve the node object for the next node
                nd.P = CurrentP                    # update that node's pressure
        kn = self.getNode(knownNode)               # get the known reference node
        deltaP = knownNodeP - kn.P                 # difference between desired and computed
        for n in self.nodes:                       # shift all nodes by deltaP
            n.P += deltaP

    def getPipe(self, name):  # Retrieve a pipe by its name string
        """
        Retrieves a pipe object by its name identifier (e.g., 'a-b').
        """
        for p in self.pipes:
            if name == p.Name():
                return p

    def getNodePipes(self, node):  # Retrieve pipes connected to a specific node
        """
        Finds and returns all pipes that connect to a specific node.
        """
        l = []
        for p in self.pipes:
            if p.oContainsNode(node):
                l.append(p)
        return l

    def nodeBuilt(self, node):  # Check if node object is already created
        """
        Checks whether a node with the given name already exists in the network.
        """
        for n in self.nodes:
            if n.name == node:
                return True
        return False

    def getNode(self, name):  # Return node object with matching name
        """
        Retrieves the node object corresponding to a given name.
        """
        for n in self.nodes:
            if n.name == name:
                return n

    def buildNodes(self):  # Automatically create node objects from pipe endpoints
        """
        Automatically constructs node objects from the pipe endpoints, ensuring each is created only once.
        """
        for p in self.pipes:  # For each pipe in the network
            if not self.nodeBuilt(p.startNode):  # If startNode doesn't exist
                self.nodes.append(Node(p.startNode, self.getNodePipes(p.startNode)))  # Create and append
            if not self.nodeBuilt(p.endNode):  # If endNode doesn't exist
                self.nodes.append(Node(p.endNode, self.getNodePipes(p.endNode)))      # Create and append

    def printPipeFlowRates(self, SI=True):  # Print flow rates for each pipe
        """
        Iterates over all pipes, printing current flow rates in the desired unit system.
        """
        for p in self.pipes:
            p.printPipeFlowRate(SI=SI)

    def printNetNodeFlows(self, SI=True):  # Print net flows for each node
        """
        Prints each node's net flow to help verify mass balance in the system.
        """
        for n in self.nodes:
            Q = n.QNet if SI else n.QNet * UC.L_to_ft3
            units = 'L/S' if SI else 'cfs'
            print('net flow into node {} is {:0.2f} ({})'.format(n.name, Q, units))

    def printLoopHeadLoss(self, SI=True):  # Print head loss for each loop
        """
        Prints the total head loss around each loop, helping verify zero net pressure drop constraint.
        """
        cf = UC.m_to_psi(1, self.pipes[0].fluid.rho)  # convert 1 m of water to psi (for English)
        units = 'm of water' if SI else 'psi'
        for l in self.loops:
            hl = l.getLoopHeadLoss()        # get head loss in meters
            hl = hl if SI else hl * cf      # convert if using English
            print('head loss for loop {} is {:0.2f} ({})'.format(l.name, hl, units))

    def printPipeHeadLoss(self, SI=True):  # Print frictional head loss for each pipe
        """
        Prints friction-related head loss in each pipe, giving a detailed breakdown of distribution losses.
        """
        for p in self.pipes:
            p.printPipeHeadLoss(SI=SI)

    def printNodePressures(self, SI=True):  # Print node pressures
        """
        Prints the computed pressure (head) at each node in the system, after referencing a known node pressure.
        """
        pUnits = 'm of water' if SI else 'psi'
        cf = 1.0 if SI else UC.m_to_psi(1, self.Fluid.rho)  # scale factor for converting from meters to psi
        for n in self.nodes:
            p = n.P * cf
            print('Pressure at node {} = {:0.2f} {}'.format(n.name, p, pUnits))


# endregion

# region function definitions
def main():  # Define main entry point
    '''
    This program analyzes flows in a given pipe network based on the following:
    1. The pipe segments are named by their endpoint node names:  e.g., a-b, b-e, etc. (see problem statement)
    2. Flow from the lower letter to the higher letter of a pipe is considered positive.
    3. Pressure decreases in the direction of flow through a pipe.
    4. At each node in the pipe network, mass is conserved.
    5. For any loop in the pipe network, the pressure loss is zero
    Approach to analyzing the pipe network:
    Step 1: build a pipe network object that contains pipe, node, loop and fluid objects
    Step 2: calculate the flow rates in each pipe using fsolve
    Step 3: output results
    Step 4: check results against expected properties of zero head loss around a loop and mass conservation at nodes.
    :return:
    '''

    SIUnits = False                         # Flag indicating usage of English units
    water = Fluid(mu=20.50e-6, rho=62.3, SI=SIUnits)  # Create Fluid object for water in English units
    r_CI = 0.00085                          # Roughness for cast iron
    r_CN = 0.003                            # Roughness for concrete
    PN = PipeNetwork()                      # Instantiate a new PipeNetwork object
    PN.Fluid = water                        # Assign the fluid to the network

    # Append Pipe objects to the network (start node, end node, length, diameter, roughness, fluid, SI)
    PN.pipes.append(Pipe('a', 'b', 1000, 18, r_CN, water, SI=SIUnits))  # Pipe from a to b
    PN.pipes.append(Pipe('a', 'h', 1600, 24, r_CN, water, SI=SIUnits))  # Pipe from a to h
    PN.pipes.append(Pipe('b', 'c', 500, 18, r_CN, water, SI=SIUnits))   # Pipe from b to c
    PN.pipes.append(Pipe('b', 'e', 800, 16, r_CI, water, SI=SIUnits))   # Pipe from b to e
    PN.pipes.append(Pipe('c', 'd', 500, 18, r_CN, water, SI=SIUnits))   # Pipe from c to d
    PN.pipes.append(Pipe('c', 'f', 800, 16, r_CI, water, SI=SIUnits))   # Pipe from c to f
    PN.pipes.append(Pipe('d', 'g', 800, 16, r_CI, water, SI=SIUnits))   # Pipe from d to g
    PN.pipes.append(Pipe('e', 'f', 500, 12, r_CI, water, SI=SIUnits))   # Pipe from e to f
    PN.pipes.append(Pipe('e', 'i', 800, 18, r_CN, water, SI=SIUnits))   # Pipe from e to i
    PN.pipes.append(Pipe('f', 'g', 500, 12, r_CI, water, SI=SIUnits))   # Pipe from f to g
    PN.pipes.append(Pipe('g', 'j', 800, 18, r_CN, water, SI=SIUnits))   # Pipe from g to j
    PN.pipes.append(Pipe('h', 'i', 1000, 24, r_CN, water, SI=SIUnits))  # Pipe from h to i
    PN.pipes.append(Pipe('i', 'j', 1000, 24, r_CN, water, SI=SIUnits))  # Pipe from i to j

    PN.buildNodes()  # Build node objects automatically by scanning pipe endpoints

    # Set external flows (cfs) at various nodes
    PN.getNode('h').setExtFlow(10, SI=SIUnits)  # Node h has +10 cfs inflow
    PN.getNode('e').setExtFlow(-3, SI=SIUnits)  # Node e has -3 cfs outflow
    PN.getNode('f').setExtFlow(-5, SI=SIUnits)  # Node f has -5 cfs outflow
    PN.getNode('d').setExtFlow(-2, SI=SIUnits)  # Node d has -2 cfs outflow

    # Define loops by listing pipes in traversal order
    PN.loops.append(
        Loop('A', [PN.getPipe('a-b'), PN.getPipe('b-e'), PN.getPipe('e-i'), PN.getPipe('h-i'), PN.getPipe('a-h')])
    )  # Loop A
    PN.loops.append(
        Loop('B', [PN.getPipe('b-c'), PN.getPipe('c-f'), PN.getPipe('e-f'), PN.getPipe('b-e')])
    )  # Loop B
    PN.loops.append(
        Loop('C', [PN.getPipe('c-d'), PN.getPipe('d-g'), PN.getPipe('f-g'), PN.getPipe('c-f')])
    )  # Loop C
    PN.loops.append(
        Loop('D', [PN.getPipe('e-f'), PN.getPipe('f-g'), PN.getPipe('g-j'), PN.getPipe('i-j'), PN.getPipe('e-i')])
    )  # Loop D

    PN.findFlowRates()  # Solve for flow rates that balance the network

    knownP = UC.psi_to_m(80, water.rho)   # Convert 80 psi to meters of fluid
    PN.getNodePressures(knownNode='h', knownNodeP=knownP)  # Set node h to correspond to 80 psi

    # Print out final results
    PN.printPipeFlowRates(SI=SIUnits)   # Flows in each pipe
    print()
    print('Check node flows:')         # Header
    PN.printNetNodeFlows(SI=SIUnits)   # Net flows at each node
    print()
    print('Check loop head loss:')     # Header
    PN.printLoopHeadLoss(SI=SIUnits)   # Loop head losses
    print()
    PN.printPipeHeadLoss(SI=SIUnits)   # Pipe head losses
    print()
    PN.printNodePressures(SI=SIUnits)  # Node pressures


# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion


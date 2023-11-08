import numpy as np
import modern_robotics as mr
import csv
import time as time_pkg


def Puppet(thetalist, dthetalist, Mlist, Slist, Glist, t, dt, springPos, restLength, damping=0, stiffness=0, g=9.81):
    '''
    Decription:
        Controls the UR5 arm using a virtual spring attached to the origin of the {b} frame. Generates csv 
            for animating in CopelliaSim
    Inputs:
        • thetalist: an n-vector of initial joint angles (units: rad)
        • dthetalist: an n-vector of initial joint rates (units: rad/s)
        • Mlist: the configurations of the link frames relative to each other at the home configuration. 
            (There are eight frames total: {0} or {s} at the base of the robot, {1} . . .{6} at the centers 
            of mass of the links, and {7} or {b} at the end-effector.)
        • Slist: the screw axes Si in the space frame when the robot is at its home configuration
        • Glist: the spatial inertia matrices Gi of the links (units: kg and kg m2)
        • t: the total simulation time (units: s)
        • dt: the simulation timestep (units: s)
        • springPos: a 3-vector indicating the location of the end of the spring not attached to the robot, 
            expressed in the {s} frame (units: m)
        • restLength: a scalar indicating the length of the spring when it is at rest (units: m)
        • damping: a scalar indicating the viscous damping at each joint (units: Nms/rad) (default = 0)
        • stiffness: a scalar indicating the stiffness of the springy string (units: N/m) (default = 0)
        • g: the gravity 3-vector in the {s} frame (units: m/s2) (default = 9.81)
    Returns:
        • thetalist_List: a list of lists of the joint angles at each time step
    '''
    
    # Intialize thetalise, dthatalist, and thetaArray (the list of lists of joint poisitions to be output)
    thetalist = thetalist
    dthetalist = dthetalist
    thetaArray = []

    # Calculate M based on Mlist
    M = []
    for m in Mlist:
        if M == []:
            M = m
        else:
            M = np.matmul(M, m)    

    time = 0.0

    # Run the simulation
    print('Starting Simulation...')
    while time <= t:
        # Calculate damping force based on joint speeds
        taulist = [i * -damping for i in dthetalist] 

        # Find the end effector location and calculate the current spring force
        tmat_ee = mr.FKinSpace(M, Slist, thetalist)
        eeRot, eePos = mr.TransToRp(tmat_ee)
        springLength = ((eePos[0] - springPos[0])**2 + (eePos[1] - springPos[1])**2 + (eePos[2] - springPos[2])**2)**0.5
        springExtension = springLength - restLength
        Fspring = springExtension * stiffness

        # Caculate the wrench on the end effector in the {b} frame
        forceTipSpace = np.array([
            Fspring * ((eePos[0] - springPos[0]) /springLength),
            Fspring * ((eePos[1] - springPos[1]) /springLength),
            Fspring * ((eePos[2] - springPos[2]) /springLength)
        ])

        forceTipBody = np.matmul(eeRot.T, forceTipSpace.T)

        Ftip = [
            0,
            0,
            0,
            forceTipBody[0],
            forceTipBody[1],
            forceTipBody[2]
        ]

        # Convert gravity to a vector
        gVector = [0, 0, -g]

        # Run ForwardDynamics function to calculate joint accelerations, then use these values, along with the current joint positions anc velocities, to find new joint positions and velocities
        ddthetalist = mr.ForwardDynamics(thetalist, dthetalist, taulist, gVector, Ftip, Mlist, Glist, Slist)
        thetalist, dthetalist = mr.EulerStep(thetalist, dthetalist, ddthetalist, dt)

        # Convert theta to be in the range -2pi to 2pi (CopelliaSim Limits)
        thetalist_mod = []
        for theta in thetalist:
            while theta > 2*np.pi:
                theta += -2*np.pi
            while theta < -2*np.pi:
                theta += 2*np.pi
            thetalist_mod.append(theta)
        thetalist = thetalist_mod

        # Add new theta list to theta array, add time step to current time
        thetaArray.append(thetalist)
        time += dt
        print(time)

    # Write thetaArray to a csv file and return it
    write_to_csv(thetaArray, 'part3b.csv')
    print('Simulation complete! Writing thetaArray to .csv file.')
    return thetaArray

def write_to_csv(array, file_name):
    """
    Decription:
        Writes an array to a csv file
    Inputs:
        • array: The array to be written to the file
        • file_name: The filename to be used
    """
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(array)


# Specify robot Mlist
M01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]])
M12 = np.array([[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0], [0, 0, 0, 1]])
M23 = np.array([[1, 0, 0, 0], [0, 1, 0, -0.1197], [0, 0, 1, 0.395], [0, 0, 0, 1]])
M34 = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.14225], [0, 0, 0, 1]])
M45 = np.array([[1, 0, 0, 0], [0, 1, 0, 0.093], [0, 0, 1, 0], [0, 0, 0, 1]])
M56 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.09465], [0, 0, 0, 1]])
M67 = np.array([[1, 0, 0, 0], [0, 0, 1, 0.0823], [0, -1, 0, 0], [0, 0, 0, 1]])
Mlist = [M01, M12, M23, M34, M45, M56, M67] 

# Specify robot Glist
G1 = np.diag([0.010267495893, 0.010267495893,  0.00666, 3.7, 3.7, 3.7])
G2 = np.diag([0.22689067591, 0.22689067591, 0.0151074, 8.393, 8.393, 8.393])
G3 = np.diag([0.049443313556, 0.049443313556, 0.004095, 2.275, 2.275, 2.275])
G4 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
G5 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
G6 = np.diag([0.0171364731454, 0.0171364731454, 0.033822, 0.1879, 0.1879, 0.1879])
Glist = [G1, G2, G3, G4, G5, G6]

# Specify robot Slist
Slist = [[0,         0,         0,         0,        0,        0],
         [0,         1,         1,         1,        0,        1],
         [1,         0,         0,         0,       -1,        0],
         [0, -0.089159, -0.089159, -0.089159, -0.10915, 0.005491],
         [0,         0,         0,         0,  0.81725,        0],
         [0,         0,     0.425,   0.81725,        0,  0.81725]]

# specify inital inputs for the Puppet function
thetalist_init = [0,0,0,0,0,0]
dthetalist_init = [0,0,0,0,0,0]
springPos = [1,1,1]
restLength = 0
simtime = 5
timeStep = 0.01

# Run Puppet function
Puppet(thetalist_init, dthetalist_init, Mlist, Slist, Glist, simtime, timeStep, springPos, restLength, damping=20, stiffness=0.0, g=9.81)
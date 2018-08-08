import numpy as np
import sys, progressbar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def get_policy(num,pos_init,tar_obs):
    M = num[0]
    T = num[1]
    Ta = num[2]
    tar1 = tar_obs[0]
    tar2 = tar_obs[1]
    obs1 = tar_obs[2]

    #state variables
    pol_mean = 6*np.ones((3,T),dtype=float)
    pol_cov = np.ones((3,3,T),dtype=float)

    last_pol_mean = np.ones((3,T),dtype=float)
    last_pol_cov = np.ones((3,3,T),dtype=float)

    #Given the initial policy
    for t in range(T):
        pol_cov[:,:,t] = 2*np.diag([1,1,1])

    #pol_mean[0,:] = np.mean(tar_obs[1][0:1])*np.ones((1,T),dtype=float)#[float(x)/T*np.mean(tar_obs[1][0:1]) for x in range(1,T+1)]
    #pol_mean[1,:] = np.mean(tar_obs[1][2:3])*np.ones((1,T),dtype=float)[float(x)/T*np.mean(tar_obs[1][2:3]) for x in range(1,T+1)]
    #pol_mean[2,:] = np.mean(tar_obs[1][4:5])*np.ones((1,T),dtype=float)#[float(x)/T*np.mean(tar_obs[1][4:5]) for x in range(1,T+1)]

    pol_mean[0,:] = [float(x)/T*np.mean(tar_obs[1][0:1]) for x in range(1,T+1)]
    pol_mean[1,:] = [float(x)/T*np.mean(tar_obs[1][2:3]) for x in range(1,T+1)]
    pol_mean[2,:] = [float(x)/T*np.mean(tar_obs[1][4:5]) for x in range(1,T+1)]

    p = progressbar.ProgressBar()
    p.start(M)

    if REAL_DIS==True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for m in range(M):
        #Sample Size at iteration m
        N = int(np.amax([np.power(m,1.1),60]))
        #Elite group size
        K = int(0.02*N)
        #sampled trajectory and control signal
        tra = np.zeros((3,T,N),dtype=float)
        con = np.zeros((3,T,N),dtype=float)
        #elite group
        elite_con = np.zeros((3,T,K),dtype=float)
        #Target1 robust degree
        J_tar1 = np.zeros((1,Ta),dtype=float)
        #Target2 robust degree
        J_tar2 = np.zeros((1,T-Ta),dtype=float)
        #obstacel robust degree
        J_obs = np.zeros((1,T),dtype=float)
        #total robustness degree
        J = np.zeros((2,N),dtype=float)
        J[0,:] = range(N)

        last_pol_mean=pol_mean
        last_pol_cov=pol_cov

        for n in range(N):
            #Sample trajectory
            tra[:,:,n],con[:,:,n] = Sample_Trajectory(pol_mean,pol_cov,pos_init,T)
            #Calculate robustness degree
            for t in range(T):
                if t<Ta:
                    J_tar1[0,t] = np.amin(np.array([tra[0,t,n]-tar1[0],tar1[1]-tra[0,t,n],tra[1,t,n]-tar1[2],tar1[3]-tra[1,t,n],tra[2,t,n]-tar1[4],tar1[5]-tra[2,t,n]]))
                else:
                    J_tar2[0,t-Ta] = np.amin(np.array([tra[0,t,n]-tar2[0],tar2[1]-tra[0,t,n],tra[1,t,n]-tar2[2],tar2[3]-tra[1,t,n],tra[2,t,n]-tar2[4],tar2[5]-tra[2,t,n]]))

                J_obs[0,t] = -np.amin(np.array([tra[0,t,n]-obs1[0],obs1[1]-tra[0,t,n],tra[1,t,n]-obs1[2],obs1[3]-tra[1,t,n],tra[2,t,n]-obs1[4],obs1[5]-tra[2,t,n]]))
            #Calculate total robustness degree
            J[1,n] = np.amin(np.array([np.amax(J_tar1[0,:]),np.amax(J_tar2[0,:]),np.amin(J_obs[0,:])]))

        #sorting the robustness degree select the elite group
        J = J[:,J[1].argsort()]
        for k in range(K):
            elite_con[:,:,k] = con[:,:,int(J[0,N-1-k])]

        con_var = np.zeros((3,3,K),dtype=float)
        #Update the parameter
        alk = 2.0/(np.power(m+150,0.501))

        for t in range(T):
            pol_mean[:,t] = alk*np.mean(elite_con[:,t,:],axis=1)+(1-alk)*last_pol_mean[:,t]
            for k in range(K):
                con_var[:,:,k] = np.outer(elite_con[:,t,k]-pol_mean[:,t],elite_con[:,t,k]-pol_mean[:,t])+0.005*np.diag([1,1,1])

            pol_cov[:,:,t] = alk*np.mean(con_var,axis=2)+(1-alk)*(last_pol_cov[:,:,t]+np.outer(last_pol_mean[:,t]-pol_mean[:,t],last_pol_mean[:,t]-pol_mean[:,t]))
        if REAL_DIS==True:

            for n in range(10):
                y,u = Sample_Trajectory(pol_mean,pol_cov,pos_init,T)
                ax.plot(y[0,:],y[1,:],y[2,:])
            plot_cube(tar_obs[0],ax,(1,0,0,0.2))
            plot_cube(tar_obs[1],ax,(0,1,0,0.2))
            plot_cube(tar_obs[2],ax,(0,0,1,1))
            #set_aspect_equal_3d(ax)
            ax.set_xlim3d(0,6)
            ax.set_ylim3d(1,4)
            ax.set_zlim3d(0,2)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.draw()
            plt.pause(0.01)
            ax.cla()

        p.update(m)
    p.finish()
    return pol_mean, pol_cov

#Plot A cube for target and Obstacle
def plot_cube(cube_pos,ax,color):
    cube_definition = [(cube_pos[0],cube_pos[2],cube_pos[4]), (cube_pos[0],cube_pos[3],cube_pos[4]), (cube_pos[1],cube_pos[2],cube_pos[4]), (cube_pos[0],cube_pos[2],cube_pos[5])]
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    faces.set_facecolor(color)

    ax.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0)

    ax.set_aspect('equal')

def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])

def plot_trajectory(pol,pos_init,tar_obs):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for n in range(10):
        y,u = Sample_Trajectory(pol[0],pol[1],pos_init,pol[0].shape[1])
        ax.plot(y[0,:],y[1,:],y[2,:])
    plot_cube(tar_obs[0],ax,(1,0,0,0.2))
    plot_cube(tar_obs[1],ax,(0,1,0,0.2))
    plot_cube(tar_obs[2],ax,(0,0,1,1))

    set_aspect_equal_3d(ax)
    # ax.set_xlim3d(0,6)
    # ax.set_ylim3d(1,4)
    # ax.set_zlim3d(0,2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    #fig.savefig('myimage.eps', format='eps', dpi=1200)

def Sample_Trajectory(theta,sigma,pos_init,T):
    ts = 0.01
    x = np.zeros((6,T),dtype=float)
    y = np.zeros((3,T),dtype=float)
    u = np.zeros((3,T),dtype=float)
    omega = 10.0
    kesai = 0.7
    o2 = omega*omega
    ko = kesai*omega
    x[:,0] = np.array([pos_init[0],0,pos_init[1],0,pos_init[2],0])
    A=np.array([[0,1,0,0,0,0],[-o2,-2*ko,0,0,0,0],[0,0,0,1,0,0],[0,0,-o2,-2*ko,0,0],[0,0,0,0,0,1],[0,0,0,0,-o2,-2*ko]])
    B = np.array([[0,0,0],[o2,0,0],[0,0,0],[0,o2,0],[0,0,0],[0,0,o2]])
    C = np.array([[1,0,0,0,0,0],[0,0,1,0,0,0],[0,0,0,0,1,0]])
    y[:,0] = C.dot(x[:,0]);
    for t in range(T-1):
        #pass
        u[:,t] = np.random.multivariate_normal(theta[:,t],sigma[:,:,t])
        x[:,t+1] = x[:,t]+ts*(A.dot(x[:,t])+B.dot(u[:,t]));
        y[:,t+1] = C.dot(x[:,t+1])
    return y,u
#Main function to call other functions
def main():
    #Numbers of Iteration
    Itera_num= 200
    #Total time horizon
    Total_hor_num = 50
    #Task a time horizon
    Task_hor_num = 20

    Num = [Itera_num,Total_hor_num,Task_hor_num]

    #Target position xl xu yl yu zl zu
    Target1_pos = np.array([1.3,1.5,2.0,2.2,0.0,0.2])
    Target2_pos = np.array([4.9,5.1,2.9,3.1,0.9,1.1])
    Obstacle_pos = np.array([2.3,3.7,2.3,3.7,0.0,1.4])

    Tar_Obs = [Target1_pos,Target2_pos,Obstacle_pos]

    #Define the initial position [x0 y0 z0]
    Init_pos = np.array([0,3,1])

    #Get the optimal policy
    policy = get_policy(Num,Init_pos,Tar_Obs)

    #plot target and obstacle
    plot_trajectory(policy,Init_pos,Tar_Obs)

if __name__ == "__main__":
    if (len(sys.argv)==2) and (sys.argv[1]=='real'):
        REAL_DIS = True
    else:
        REAL_DIS = False
    main()

using Finite element analysis>>
@author of the code: Tong Wu
Contact: wu616@purdue.edu

The problem is shown in << Fundamentals of the finite 
element method for heat and fluid flow>>
Author: Roland W Lewis, Perumal Nithiarasu 
and Kankanhally N. Seetharamu
pp108-pp112 and pp172

|node 1------node 2------node 3
        bar1        bar2
ambient temperature 25 C
Initial prescribed temperature, on node 1, 100 C
Other parameters are:
h=120 #convection
k=200 #thermal conductivity [W/m*C]
rho=1 #density 
P=10e-3 #perimeter [m]
L=1e-2 #length per element
A=6e-6 #section area [m^2]
C_p=2.42e6 #specific heat [W/m**3*C] 
dt=0.1 #time step
"""
def eofhtb():    
    import sympy
    from sympy import Matrix 
    import numpy as np
    import scipy
    # H_e, convection; K_e, conduction; C_e, specific heat
    h=120 #convection
    T_a=25 #ambient temperature [C]
    k=200 #thermal conductivity [W/m*C]
    rho=1 #density 
    P=10e-3 #perimeter [m]
    L=1e-2 #length per element
    A=6e-6 #section area [m^2]
    C_p=2.42e6 #specific heat [W/m**3*C]
    sympy.var('x')
    N=Matrix([[1-x], [x]]) #shape function
    B=Matrix([-1,1]) #temperature-gradient function
    f_e=Matrix([1,1]) #elementwise heat flux
    f_e= (h*P*L*T_a/2)*f_e #elementwise heat flux 
    #initialize C_e, H_e, K_e
    C_ee= N*np.transpose(N) 
    H_ee= N*np.transpose(N)
    K_ee= B*np.transpose(B)
    x1,x2=(0,1)    
    from sympy.utilities import lambdify
    from mpmath import quad
    C_e=scipy.zeros(C_ee.shape, dtype=float)
    H_e=scipy.zeros(H_ee.shape, dtype=float)
    K_e=scipy.zeros(K_ee.shape, dtype=float)
    #Numerical integration
    for (i,j), expr1 in scipy.ndenumerate(H_ee):
        H_e[i,j] = quad(lambdify((x),expr1,'math'),(x1,x2))
        H_e[i,j] = h*P*L*H_e[i,j]
        C_e[i,j] = rho*C_p*L*A*quad(lambdify((x),expr1,'math'),(x1,x2))
    for (i,j), expr2 in scipy.ndenumerate(K_ee):    
        K_e[i,j] = quad(lambdify((x),expr2,'math'),(x1,x2))
        K_e[i,j] = ((A*k)/L)*K_e[i,j]
    #Compute total stiffness (Conduction+Convection)
    K_e_total=H_e+K_e
    return (H_e,K_e,C_e,K_e_total,f_e);

def main():
    import numpy as np
    (H_e,K_e,C_e,K_e_total,f_e)=eofhtb() #import function of stiffness matrix
    # initialize 
    C=np.zeros((3,3))   #specific heat matrix (2 bars)
    K=np.zeros((3,3))   #stiffness matrix (2 bars)
    F=np.zeros((3,1))   #external heat source 
    T_0=np.zeros((3,1)) #ambient temperature
    T_1=np.zeros((3,1)) #ambient temperature + prescribed temperature
    T_2=np.zeros((3,1)) #updagted temperature
    F_prescribed=np.zeros((3,1)) #prescribed heat source 
    #finite element analysis 
    T_0[:,0]=[25,25,25] #define the ambient temperature
    #assemble global stiffness matrix 
    for i in range(2):
        C[i:i+2, i:i+2] += C_e
        K[i:i+2, i:i+2] += K_e_total
        F[i:i+2, :] += f_e
    #set parameters of Back-Euler method
    #theta=1 #parameter theta: T^{n+theta}=theta*T^{n+1}+(1-theta)*T^{n} 
    t=0 #time
    dt=0.1 #time step 
    K_and_C_rhs=(C/dt) #the global matrix of right hand side
    K_and_C_lhs=(C/dt)+K #the global matrix of left hand side 
    F_dt=F #external matrix for discretized delta-T
    #1st iteration
    loop=0;loopmax=200
    T_1[:,0]=[100,25,25] #define the prescribed temperature for node 1.
    T_storage=np.zeros((loopmax,4))
    #iterative solver of Crank-Nicolson method
    while loop<loopmax:
        loop=loop+1
        t=t+dt
        F_dt1=np.matmul(K_and_C_rhs,T_0)+F_dt #
        F_prescribed[1:3,0]=-K_and_C_lhs[1:3,0]*T_1[0,0]
        F_dt1_update=F_dt1+F_prescribed
        F_dt_1_up_redu=F_dt1_update[1:3]
        K_and_C_lhs_reduce=K_and_C_lhs[1:3,1:3]
        T_2[1:3]=np.linalg.solve(K_and_C_lhs_reduce,F_dt_1_up_redu)
        T_2[0]=100
        T_storage[loop-1,1:4]=T_2.transpose()
        T_storage[loop-1,0]=t
        T_0=T_2;T_1=T_2
    #print(T_storage)
    from tempfile import TemporaryFile
    outfile = TemporaryFile()
    np.save(outfile,T_storage)
    #plot temperature of three nodes 
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots() 
    ax.plot(T_storage[:,0], T_storage[:,1], color="blue", label="T,node 1")
    ax.plot(T_storage[:,0], T_storage[:,2], color="red", label="T,node 2")
    ax.plot(T_storage[:,0], T_storage[:,3], color="green", label="T,node 3")
    ax.set_xlabel("time step (s)")
    ax.set_ylabel("temperature (C)")
    ax.legend()
    plt.show()
#main driver     
if __name__ == "__main__":
    main()
    

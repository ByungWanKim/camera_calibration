import numpy as np
import scipy as sp
from scipy import signal
from array import *
from Tkinter import *
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from pylab import arange, plot, sin, ginput, show
import math
import os
from numpy.linalg import inv
import copy
import numpy.matlib
import scipy.sparse as sps

###############initial_value#######
dX_default = 30
dY_default = 30
n_sq_x_default = 10
n_sq_y_default = 10
f=open("image_name_calib.txt",'r')
image_name=f.read().split()
#print(image_name)
f.close()
i=0
while i<np.size(image_name):
    exec('I_'+str(i+1)+'=cv2.imread(image_name[i])')
    exec('I_'+str(i+1)+'=cv2.cvtColor(I_'+str(i+1)+',cv2.COLOR_BGR2GRAY)')
    exec('I_'+str(i+1)+'=np.float_(I_'+str(i+1)+')')
    exec('dX_'+str(i+1)+'=None')
    exec('dY_'+str(i+1)+'=None')
    exec('wintx_'+str(i+1)+'=None')
    exec('winty_'+str(i+1)+'=None')
    exec('x_'+str(i+1)+'=None')
    exec('X_'+str(i+1)+'=None')
    exec('n_sq_x_'+str(i+1)+'=None')
    exec('n_sq_y_'+str(i+1)+'=None')
    i=i+1
n_ima=np.size(image_name)
nx=np.size(I_1,1)
ny=np.size(I_1,0)
wintx_default = max(round(nx/128),round(ny/96))
winty_default = wintx_default
ima_numbers = 0
ima_proc=[]
kk_first = 0
wintx=wintx_default
winty=winty_default
manual_squares=0
x=0
y=0
point=0


kbw=None;
kbw1=None;
#######rigid_motion##########
def rigid_motion(X,om,T):
    global kbw
    rodri_out=rodrigues(om)
    R=rodri_out[0]
    dRdom=rodri_out[1]
    m,n=X.shape
    Y=R*X+numpy.matlib.repmat(T,1,n)

    dYdR = np.mat(np.zeros((3*n,9)))
    dYdT = np.mat(np.zeros((3*n,3)))

    dYdR[0:3*n:3,0:9:3] =  X.T
    dYdR[1:3*n:3,1:9:3] =  X.T
    dYdR[2:3*n:3,2:9:3] =  X.T

    dYdT[0:3*n:3,0] =  np.mat(np.ones((n,1)))
    dYdT[1:3*n:3,1] =  np.mat(np.ones((n,1)))
    dYdT[2:3*n:3,2] =  np.mat(np.ones((n,1)))

    dYdom = dYdR * dRdom;
    
    return Y,dYdom,dYdT

########project_points2##############
def project_points2(X,om,T,f,c,k,alpha):
    m,n=X.shape
    rigid_motion_out=rigid_motion(X,om,T)
    Y=rigid_motion_out[0]
    dYdom=rigid_motion_out[1]
    dYdom_row,dYdom_col=dYdom.shape
    dYdT=rigid_motion_out[2]
    inv_Z = 1/Y[2,:]
    x=np.multiply(Y[0:2,:],(np.mat(np.ones((2,1))*inv_Z)))

    bb=(np.multiply(-x[0,:],inv_Z).T)*np.mat(np.ones((1,3)))
    cc=(np.multiply(-x[1,:],inv_Z).T)*np.mat(np.ones((1,3)))
    dxdom = np.mat(np.zeros((2*n,3)))
    dxdom[0:2*n:2,:]=np.multiply(((inv_Z.T)*np.mat(np.ones((1,3)))),dYdom[0:dYdom_row:3,:])+np.multiply(bb,dYdom[2:dYdom_row:3,:])
    dxdom[1:2*n:2,:]=np.multiply(((inv_Z.T)*np.mat(np.ones((1,3)))),dYdom[1:dYdom_row:3,:])+np.multiply(cc,dYdom[2:dYdom_row:3,:])

    dxdT = np.mat(np.zeros((2*n,3)))
    dxdT[0:2*n:2,:]=np.multiply(((inv_Z.T)*np.mat(np.ones((1,3)))),dYdT[0:dYdom_row:3,:])+np.multiply(bb,dYdT[2:dYdom_row:3,:])
    dxdT[1:2*n:2,:]=np.multiply(((inv_Z.T)*np.mat(np.ones((1,3)))),dYdT[1:dYdom_row:3,:])+np.multiply(cc,dYdT[2:dYdom_row:3,:])

    r2=np.square(x[0,:])+np.square(x[1,:])
    r2_row,r2_col=r2.shape
    dr2dom=np.multiply((2*(x[0,:].T)*np.mat(np.ones((1,3)))),dxdom[0:2*n:2,:])+np.multiply((2*(x[1,:].T)*np.mat(np.ones((1,3)))),dxdom[1:2*n:2,:])
    dr2dT=np.multiply((2*(x[0,:].T)*np.mat(np.ones((1,3)))),dxdT[0:2*n:2,:])+np.multiply((2*(x[1,:].T)*np.mat(np.ones((1,3)))),dxdT[1:2*n:2,:])

    r4=np.square(r2)
    r4_row,r4_col=r4.shape
    dr4dom=np.multiply((2*(r2.T)*np.mat(np.ones((1,3)))),dr2dom)
    dr4dT=np.multiply((2*(r2.T)*np.mat(np.ones((1,3)))),dr2dT)

    r6=np.power(r2,3)
    r6_row,r6_col=r6.shape
    dr6dom=np.multiply((3*np.square(r2.T)*np.mat(np.ones((1,3)))),dr2dom)
    dr6dT=np.multiply((3*np.square(r2.T)*np.mat(np.ones((1,3)))),dr2dT)

    cdist=1+k[0,0]*r2+k[1,0]*r4+k[4,0]*r6

    dcdistdom = k[0,0] * dr2dom + k[1,0] * dr4dom + k[4,0] * dr6dom
    dcdistdT = k[0,0] * dr2dT + k[1,0] * dr4dT + k[4,0] * dr6dT
    
    dcdistdk = np.mat(np.zeros((n,r2_row+r4_row+2+r6_row)))
    dcdistdk[:,0:r2_row]=r2.T
    dcdistdk[:,r2_row:r2_row+r4_row]=r4.T
    dcdistdk[:,r2_row+r4_row+2:r2_row+r4_row+2+r6_row]=r6.T

    xd1=np.multiply(x,(np.mat(np.ones((2,1)))*cdist))

    dxd1dom =np.mat(np.zeros((2*n,3)))
    dxd1dom[0:2*n:2,:]=np.multiply((x[0,:].T)*np.mat(np.ones((1,3))),dcdistdom)
    dxd1dom[1:2*n:2,:]=np.multiply((x[1,:].T)*np.mat(np.ones((1,3))),dcdistdom)
    cdist_row,cdist_col=cdist.shape
    coeff_temp=np.mat(np.zeros((cdist_row*2,cdist_col)))
    coeff_temp[0:cdist_row,:]=cdist
    coeff_temp[cdist_row:2*cdist_row,:]=cdist
    coeff=((coeff_temp.T).reshape(2*n,1))*np.mat(np.ones((1,3)))
    dxd1dom=dxd1dom+np.multiply(coeff,dxdom)

    dxd1dT =np.mat(np.zeros((2*n,3)))
    dxd1dT[0:2*n:2,:]=np.multiply((x[0,:].T)*np.mat(np.ones((1,3))),dcdistdT)
    dxd1dT[1:2*n:2,:]=np.multiply((x[1,:].T)*np.mat(np.ones((1,3))),dcdistdT)
    dxd1dT=dxd1dT+np.multiply(coeff,dxdT)

    dxd1dk =np.mat(np.zeros((2*n,5)))
    dxd1dk[0:2*n:2,:]=np.multiply((x[0,:].T)*np.mat(np.ones((1,5))),dcdistdk)
    dxd1dk[1:2*n:2,:]=np.multiply((x[1,:].T)*np.mat(np.ones((1,5))),dcdistdk)

    a1=2*np.multiply(x[0,:],x[1,:])
    a2=r2+2*(np.square(x[0,:]))
    a3=r2+2*(np.square(x[1,:]))

    delta_x_row1=k[2,0]*a1+k[3,0]*a2
    delta_x_row1_row,delta_x_row1_col=delta_x_row1.shape
    delta_x_row2=k[2,0]*a3+k[3,0]*a1
    delta_x_row2_row,delta_x_row2_col=delta_x_row2.shape
    delta_x=np.mat(np.zeros((delta_x_row1_row+delta_x_row2_row,delta_x_row1_col)))
    delta_x[0:delta_x_row1_row,:]=delta_x_row1
    delta_x[delta_x_row1_row:delta_x_row1_row+delta_x_row2_row,:]=delta_x_row2

    aa = ((2*k[2,0]*x[1,:]+6*k[3,0]*x[0,:]).T)*np.mat(np.ones((1,3)))
    bb = ((2*k[2,0]*x[0,:]+2*k[3,0]*x[1,:]).T)*np.mat(np.ones((1,3)))
    cc = ((6*k[2,0]*x[1,:]+2*k[3,0]*x[0,:]).T)*np.mat(np.ones((1,3)))

    ddelta_xdom = np.mat(np.zeros((2*n,3)))        
    ddelta_xdom[0:2*n:2,:]=np.multiply(aa,dxdom[0:2*n:2,:])+np.multiply(bb,dxdom[1:2*n:2,:])
    ddelta_xdom[1:2*n:2,:]=np.multiply(bb,dxdom[1:2*n:2,:])+np.multiply(cc,dxdom[1:2*n:2,:])

    ddelta_xdT = np.mat(np.zeros((2*n,3)))
    dxdT_row,dxdT_col=dxdT.shape
    ddelta_xdT[0:2*n:2,:]=np.multiply(aa,dxdT[0:2*n:2,:])+np.multiply(bb,dxdT[1:2*n:2,:])
    ddelta_xdT[1:2*n:2,:]=np.multiply(bb,dxdT[1:2*n:2,:])+np.multiply(cc,dxdT[1:2*n:2,:])

    ddelta_xdk = np.mat(np.zeros((2*n,5)))
    ddelta_xdk[0:2*n:2,2] = a1.T
    ddelta_xdk[0:2*n:2,3] = a2.T
    ddelta_xdk[1:2*n:2,2] = a3.T
    ddelta_xdk[1:2*n:2,3] = a1.T

    xd2=xd1+delta_x
    xd2_row,xd2_col=xd2.shape

    dxd2dom = dxd1dom + ddelta_xdom 
    dxd2dT = dxd1dT + ddelta_xdT
    dxd2dk = dxd1dk + ddelta_xdk 

    
    xd3_row1=xd2[0,:]+alpha*xd2[1,:]
    xd3_row2=xd2[1,:]
    xd3=np.mat(np.zeros((2,xd2_col)))
    xd3[0,:]=xd3_row1
    xd3[1,:]=xd3_row2
    
    
    dxd3dom = np.mat(np.zeros((2*n,3)))
    dxd3dom[0:2*n:2,:] = dxd2dom[0:2*n:2,:] + alpha*dxd2dom[1:2*n:2,:]
    dxd3dom[1:2*n:2,:] = dxd2dom[1:2*n:2,:]
    dxd3dT = np.mat(np.zeros((2*n,3)))
    dxd3dT[0:2*n:2,:] = dxd2dT[0:2*n:2,:] + alpha*dxd2dT[1:2*n:2,:]
    dxd3dT[1:2*n:2,:] = dxd2dT[1:2*n:2,:]
    dxd3dk = np.mat(np.zeros((2*n,5)))
    dxd3dk[0:2*n:2,:] = dxd2dk[0:2*n:2,:] + alpha*dxd2dk[1:2*n:2,:]
    dxd3dk[1:2*n:2,:] = dxd2dk[1:2*n:2,:]
    dxd3dalpha = np.mat(np.zeros((2*n,1)))
    dxd3dalpha[0:2*n:2,:] = xd2[1,:].T

    f_row,f_col=f.shape
    c_row,c_col=c.shape
    f_size=np.size(f)
    c_size=np.size(c)
    f_re=(f.T).reshape(f_size,1)
    c_re=(c.T).reshape(c_size,1)
    
    if len(f)>1:
        xp=np.multiply(xd3,(f_re*np.mat(np.ones((1,n)))))+c_re*np.mat(np.ones((1,n)))
        coeff_temp=f_re*np.mat(np.ones((1,n)))
        coeff=(coeff_temp.T).reshape(2*n,1)
        dxpdom = np.multiply((coeff*np.mat(np.ones((1,3)))), dxd3dom)
        dxpdT = np.multiply((coeff*np.mat(np.ones((1,3)))), dxd3dT)
        dxpdk = np.multiply((coeff*np.mat(np.ones((1,5)))), dxd3dk)
        dxpdalpha = np.multiply(coeff,dxd3dalpha)
        dxpdf = np.mat(np.zeros((2*n,2)))
        dxpdf[0:2*n:2,0] = xd3[0,:].T
        dxpdf[1:2*n:2,1] = xd3[1,:].T
    else:
        xp=f*xd3+c*np.mat(np.ones(1,n))
        dxpdom = f * dxd3dom
        dxpdT = f * dxd3dT
        dxpdk = f  * dxd3dk
        dxpdalpha = np.multiply(f,dxd3dalpha)
        
        dxpdf = (xd3.T).reshape(xd2_col,1)

    dxpdc=np.mat(np.zeros((2*n,2)))
    dxpdc[0:2*n:2,0] = np.mat(np.ones((n,1)))
    dxpdc[1:2*n:2,1] = np.mat(np.ones((n,1)))

    
    return xp,dxpdom,dxpdT,dxpdf,dxpdc,dxpdk,dxpdalpha

    

    

#########compute_extrinsic_refine##############
def compute_extrinsic_refine(omc_init,Tc_init,x_kk,X_kk,fc,cc,kc,alpha_c,MaxIter,thresh_cond):
    omckk=omc_init
    Tckk=Tc_init
    param=np.mat(np.ones((6,1)))
    param[0:3,0]=omckk
    param[3:6,0]=Tckk
    change=1
    Iter=0
    while (change>(1e-10))&(Iter<MaxIter):
        project_point_out=project_points2(X_kk,omckk,Tckk,fc,cc,kc,alpha_c)
        x=project_point_out[0]
        dxdom=project_point_out[1]
        dxdT=project_point_out[2]

        ex=x_kk-x
        ex_size=np.size(ex)
        dxdom_row,dxdom_col=dxdom.shape
        dxdT_row,dxdT_col=dxdT.shape
        JJ = np.mat(np.zeros((dxdom_row,dxdom_col+dxdT_col)))
        JJ[:,0:dxdom_col]=dxdom
        JJ[:,dxdom_col:dxdom_col+dxdT_col]=dxdT
        if np.linalg.cond(JJ)>thresh_cond:
            change=0
        else:
            JJ2=JJ.T*JJ

            param_innov=inv(JJ2)*(JJ.T)*(ex.T).reshape(ex_size,1)
            param_up=param+param_innov
            change=np.linalg.norm(param_innov)/np.linalg.norm(param_up)
            param=param_up
            Iter=Iter+1

            omckk=param[0:3]
            Tckk=param[3:6]
    
    rodri_out=rodrigues(omckk)
    Rckk=rodri_out[0]
    return omckk,Tckk,Rckk,JJ

#########rodrigues#############
def rodrigues(IN):
    global kbw
    m,n=IN.shape
    
    bigeps=10e+20*np.spacing(1)
    if ((m==1)&(n==3))|((m==3)&(n==1)):
        theta=np.linalg.norm(IN)
        if theta<np.spacing(1):
            R=np.eye(3)

            dRdin=np.mat([[0,0,0],[0,0,1],[0,-1,0],[0,0,-1],[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,0,0]])
        else:
            if n==len(IN): IN=IN.T
            dm3din = np.mat(np.zeros((4,3)))
            dm3din[0:3,:]=np.eye(3)
            dm3din[3,:]=IN.T/theta
            omega=IN/theta
            dm2dm3=np.mat(np.zeros((4,4)))
            dm2dm3[0:3,0:3]=np.eye(3)/theta
            dm2dm3[3,3]=1
            dm2dm3[0:3,3]=-IN/(theta**2)
            alpha=math.cos(theta)
            beta=math.sin(theta)
            gamma=1-math.cos(theta)
            omegav=np.mat([[0,-omega[2,0],omega[1,0]],[omega[2,0],0,-omega[0,0]],[-omega[1,0],omega[0,0],0]])
            A=omega*omega.T
            dm1dm2=np.mat(np.zeros((21,4)))
            dm1dm2[0,3]=-math.sin(theta)
            dm1dm2[1,3]=math.cos(theta)
            dm1dm2[2,3]=math.sin(theta)
            dm1dm2[3:12,0:3]=np.mat([[0,0,0,0,0,1,0,-1,0],[0,0,-1,0,0,0,1,0,0],[0,1,0,-1,0,0,0,0,0]]).T
            w1=omega[0,0]
            w2=omega[1,0]
            w3=omega[2,0]
            
            dm1dm2[12:21,0] = np.mat([[2*w1],[w2],[w3],[w2],[0],[0],[w3],[0],[0]])
            dm1dm2[12:21,1] = np.mat([[0],[w1],[0],[w1],[2*w2],[w3],[0],[w3],[0]])
            dm1dm2[12:21,2] = np.mat([[0],[0],[w1],[0],[0],[w2],[w1],[w2],[2*w3]])

            R=np.eye(3)*alpha+omegav*beta+A*gamma

            dRdm1 = np.mat(np.zeros((9,21)))
            
            dRdm1[[0,4,8],0] = 1
            omegav_size=np.size(omegav)
            dRdm1[:,1] = (omegav.T).reshape(omegav_size,1)
            dRdm1[:,3:12] = beta*np.eye(9)
            dRdm1[:,2] = (A.T).reshape(omegav_size,1)
            dRdm1[:,12:21] = gamma*np.eye(9)

            dRdin = dRdm1 * dm1dm2 * dm2dm3 * dm3din

        R=np.mat(R)
        dRdin=np.mat(dRdin)
        out = R.copy()
        dout = dRdin.copy()



    elif (m==n)&(m==3)&(np.linalg.norm((IN.T)*(IN-np.eye(3)))<bigeps)&((abs(np.linalg.det(IN)-1))<bigeps):
        R=IN.copy()
    
        U,S,V=np.linalg.svd(R)
        V=V.T
        R=U*(V.T)
        tr = (np.trace(R)-1)/2
        dtrdR = np.mat([1., 0., 0., 0., 1., 0., 0. ,0., 1.])/2
        theta=np.real(np.arccos(tr))
        if np.sin(theta)>=(1e-4):
            dthetadtr = -1/np.sqrt(1-tr**2)
            dthetadR = dthetadtr * dtrdR
            vth = 1/(2*np.sin(theta))
            dvthdtheta = -vth*np.cos(theta)/np.sin(theta)
            dvar1dtheta = np.mat([[dvthdtheta],[1]])

            dvar1dR =  dvar1dtheta * dthetadR


            om1 = np.mat([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]).T

            dom1dR = np.mat([[0,0,0,0,0,1,0,-1,0],[0,0,-1,0,0,0,1,0,0],[0,1,0,-1,0,0,0,0,0]])
            dom1dR_row=dom1dR.shape[0]
            dom1dR_col=dom1dR.shape[1]
            dvar1dR_row=dvar1dR.shape[0]
            dvardR=np.mat(np.zeros((dom1dR_row+dvar1dR_row,dom1dR_col)))
            dvardR[0:dom1dR_row,:] = dom1dR
            dvardR[dom1dR_row:dom1dR_row+dvar1dR_row,:]=dvar1dR

            om = vth*om1
            domdvar = np.mat(np.zeros((3,5)))
            domdvar[:,0:3] = vth*np.eye(3)
            domdvar[:,3] = om1
            dthetadvar = np.mat([0,0,0,0,1])
            
            dvar2dvar=np.mat(np.zeros((4,5)))
            dvar2dvar[0:3,:]=domdvar
            dvar2dvar[3,:]=dthetadvar
            

            out = om*theta
            domegadvar2 = np.mat(np.zeros((3,4)))
            domegadvar2[:,0:3] = theta*np.eye(3)
            domegadvar2[:,3] = om
            

            dout = domegadvar2 * dvar2dvar * dvardR
            
        else:
            if tr>0:
                out=np.mat([0,0,0]).T
                dout=np.mat([[0,0,0,0,0,1/2,0,-1/2,0],[0,0,-1/2,0,0,0,1/2,0,0],[0,1/2,0,-1/2,0,0,0,0,0]])
            else:
                hashvec =np.mat([0,-1,-3,-9,9,3,1,13,5,-7,-11]).T
                Smat=np.mat([[1,1,1],[1,0,-1],[0,1,-1],[1,-1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1],[1,1,-1],[1,-1,-1],[1,-1,1]])
                M = (R+np.eye(3))/2
                uabs = np.sqrt(M[0,0])
                vabs = np.sqrt(M[1,1])
                wabs = np.sqrt(M[2,2])

                mvec=np.mat([M[0,2]+M[1,0],M[0,2]+M[2,1],M[0,2]+M[2,0]])/2
                syn = (mvec>np.spacing(1))-(mvec<-np.spacing(1))
                Hash = syn*(np.mat([9,3,1]).T)
                idx=np.where(Hash==hashvec)[0][0,0]
                svec = Smat[idx,:].T

                out=theta*np.multiply((np.mat([uabs,vabs,wabs]).T),svec)
    else:
        print('Neither a rotation matrix nor a rotation vector were provided')

    return out,dout
##########normalize_pixel##############
def normalize_pixel(x_kk,fc,cc,kc,alpha_c):
    x_distort=np.mat(np.zeros((2,x_kk.shape[1])))
    x_distort[0,:] = (x_kk[0,:]-cc[0,0])/fc[0,0]
    x_distort[1,:] = (x_kk[1,:]-cc[1,0])/fc[1,0]

    x_distort[0,:]=x_distort[0,:]-alpha_c*x_distort[1,:]

    if np.linalg.norm(kc) !=0:
        k1=kc[0]
        k2=kc[1]
        k3=kc[4]
        p1=kc[2]
        p2=kc[3]
        x=x_distort.copy()
        
        for kk in range(1,21):
            r_2=sum(np.square(x))
            k_radial =  1 + k1 * r_2 + k2 * np.square(r_2) + k3 * np.power(r_2,3)
            delta_x1=2*p1*np.multiply(x_distort[0,:],x_distort[1,:]) + p2*(r_2 + 2*np.square(x_distort[0,:]))
            delta_x2=p1*(r_2 + 2*np.square(x_distort[1,:]))+2*p2*np.multiply(x_distort[0,:],x_distort[1,:])
            delta_x1_row,delta_x1_col=delta_x1.shape
            delta_x2_row,delta_x2_col=delta_x2.shape
                     
            delta_x=np.mat(np.zeros((delta_x1_row+delta_x2_row,delta_x1_col)))

            delta_x[0:delta_x1_row,:]=delta_x1
            delta_x[delta_x1_row:delta_x1_row+delta_x2_row,:]=delta_x2
            
            x = np.multiply((x_distort - delta_x),1/((np.mat(np.ones((2,1)))*k_radial)))
        xn=x.copy()
    else:
        xn=x_distort.copy()
    
    return xn
    
###########compute_extrinsic_init############
def compute_extrinsic_init(x_kk,X_kk,fc,cc,kc,alpha_c):
    global kbw
    ###normalize_pixel###
    xn=normalize_pixel(x_kk,fc,cc,kc,alpha_c)
    Np=xn.shape[1]
    X_mean=np.mean(X_kk,axis=1)
    Y=X_kk-(X_mean*np.mat(np.ones((1,Np))))
    YY=Y*Y.T
    U,S,V=np.linalg.svd(YY)
    V=V.T
    r=S[2]/S[1]

    if (r<1e-3)|(Np<5):
        R_transform = V.T
        if np.linalg.norm(R_transform[0:2,2])<1e-6:
            R_transform=np.eye(3)
        if np.linalg.det(R_transform)<0:
            R_transform=-R_transform
        
        T_transform = -(R_transform)*X_mean
        X_new = R_transform*X_kk + T_transform*np.mat(np.ones((1,Np)))
        homography_out = compute_homography(xn,X_new[0:2,:])
        H=homography_out[0].copy()
        sc=np.mean(np.mat([[np.linalg.norm(H[:,0])],[np.linalg.norm(H[:,1])]]),axis=0)
        H=H/sc
        u1=H[:,0]
        u1=u1/np.linalg.norm(u1)
        u2=H[:,1]-np.dot(u1.T,H[:,1])[0,0]*u1
        u2=u2/np.linalg.norm(u2)
        u3=np.cross(u1.T,u2.T).T
        u1_row=u1.shape[0]
        u1_col=u1.shape[1]
        u2_col=u2.shape[1]
        u3_col=u3.shape[1]
        RRR=np.mat(np.ones((u1_row,u1_col+u2_col+u3_col)))
        RRR[:,0:u1_col]=u1
        RRR[:,u1_col:u1_col+u2_col]=u2
        RRR[:,u1_col+u2_col:u1_col+u2_col+u3_col]=u3
        
        rodri_out=rodrigues(RRR)
        
        omckk=rodri_out[0]
        
        rodri_out=rodrigues(omckk)
        Rckk=rodri_out[0]
        
        Tckk=H[:,2]
        Tckk = Tckk + Rckk* T_transform
        Rckk = Rckk * R_transform
        
        rodri_out=rodrigues(Rckk)
        omckk=rodri_out[0]
        rodri_out=rodrigues(omckk)
        Rckk=rodri_out[0]
    else:
        J=np.mat(np.zeros((2*Np,12)))
        xX=np.multiply(np.mat(np.ones((3,1)))*xn[0,:],X_kk)
        yX=np.multiply(np.mat(np.ones((3,1)))*xn[1,:],X_kk)

        J[0:2*Np:2,[0,3,6]]=-X_kk.T
        J[1:2*Np:2,[1,4,7]]=X_kk.T
        J[0:2*Np:2,[2,5,8]]=xX.T
        J[1:2*Np:2,[2,5,8]]=-yX.T
        J[0:2*Np:2,11]=xn[0,:].T
        J[1:2*Np:2,11]=-xn[1,:].T
        J[0:2*Np:2,9]=-np.ones((Np,1))
        J[1:2*Np:2,10]=np.ones((Np,1))

        JJ = J.T*J
        U,S,V=np.linalg.svd(JJ)
        V=V.T
        RR=(V[0:9,11].T).reshape(3,3)
        if np.linalg.det(RR)<0:
            V[:,11]=-V[:,11]
            RR=-RR

        Ur,Sr,Vr=np.linalg.svd(RR)
        Rckk = Ur*Vr
        sc=np.linalg.norm(V[0:9,11])/np.linalg.norm(Rckk)
        Tckk=V[9:12,11]/sc
        rodri_out=rodrigues(Rckk)
        omckk=rodri_out[0]
        rodri_out=rodrigues(omckk)
        Rckk=rodri_out[0]
    return omckk,Tckk,Rckk
###########go_calib_optim####################
def go_calib_optim():
    global kc
    global kbw
    global kbw1
    with open('objs.pickle') as f: objs= pickle.load(f)

    active_images=objs[0]
    ind_active=objs[1]
    wintx=objs[2]
    winty=objs[3]
    n_ima=objs[4]
    nx=objs[5]
    ny=objs[6]
    dX_default=objs[7]
    dY_default=objs[8]
    dX=objs[9]
    dY=objs[10]
    wintx_default=objs[11]
    winty_default=objs[12]
    for kk in np.arange(0,n_ima):
        exec('X_'+str(kk+1)+'=objs[kk*8+13]') in locals(),globals()
        exec('x_'+str(kk+1)+'=objs[kk*8+14]') in locals(),globals()
        exec('n_sq_x_'+str(kk+1)+'=objs[kk*8+15]') in locals(),globals()
        exec('n_sq_y_'+str(kk+1)+'=objs[kk*8+16]') in locals(),globals()
        exec('wintx_'+str(kk+1)+'=objs[kk*8+17]') in locals(),globals()
        exec('winty_'+str(kk+1)+'=objs[kk*8+18]') in locals(),globals()
        exec('dX_'+str(kk+1)+'=objs[kk*8+19]') in locals(),globals()
        exec('dY_'+str(kk+1)+'=objs[kk*8+20]') in locals(),globals()

    desactivated_images=[]
    recompute_extrinsic=(len(ind_active)<100)
    rosette_calibration = 0
    if rosette_calibration:
        est_dist=np.float_(np.ones((5,1)))

    ###go_calib_optim_iter###
    est_aspect_ratio = 1.
    est_fc = np.mat([[1.],[1.]])
    MaxIter = 30
    check_cond = 1
    center_optim = 1.
    
    if ('est_dist' not in globals().keys())&('est_dist' not in locals().keys()):
        est_dist=np.mat([[1],[1],[1],[1],[0]])
    else:
        if len(est_dist)==4:
            est_dist_temp=np.zeros((5,1))
            est_dist_temp[0:4,:]=est_dist
            est_dist=est_dist_temp

    if ('est_alpha' not in globals().keys())&('est_alpha' not in locals().keys()):
        est_alpha = 0.



    quick_init = 0
    dont_ask=0
    alpha_c = 0

    if np.prod(est_dist)==0:
        print 'Distortion not fully estimated (defined by the variable est_dist):\n'
        if est_dist[0,0]==0:
            print '     Second order distortion not estimated (est_dist(1)=0).\n'
        if est_dist[1,0]==0:
            print '     Fourth order distortion not estimated (est_dist(2)=0).\n'
        if est_dist[4,0]==0:
            print '     Sixth order distortion not estimated (est_dist(5)=0) - (DEFAULT) .\n'
        if np.prod(est_dist[2:4,0])==0:
            print '     Tangential distortion not estimated (est_dist(3:4)~=[1;1]).\n'
     
    alpha_smooth = 0.1
    thresh_cond = 1e6
    if ('cc' not in globals().keys())&('cc' not in locals().keys()):
        cc = np.mat([[(nx-1)/2],[(ny-1)/2]])
        alpha_smooth=0.1
        
    ###init_intrinsic_param###    
    two_focals_init=0.

    for kk in range(1,n_ima+1):
        exec('x_kk=x_'+str(kk)+'.copy()')
        exec('X_k=X_'+str(kk)+'.copy()')
        if active_images[kk-1]:
            exec('Hout=compute_homography(x_kk,X_k[0:2,:])')
            exec('H_'+str(kk)+'=Hout[0]')

    c_init=(np.float_(np.mat([[nx],[ny]]))/2)-0.5        
    k_init=np.mat([[0.],[0.],[0.],[0.],[0.]])
    A=[]
    b=[]
    Sub_cc=np.mat([[1,0,-c_init[0]],[0,1,-c_init[1]],[0,0,1]])

    for kk in range(1,n_ima+1):
        if active_images[kk-1]:
            exec('Hkk=H_'+str(kk)+'.copy()')
            Hkk=Sub_cc*Hkk

            V_hori_pix=Hkk[:,0]
            V_vert_pix=Hkk[:,1]
            V_diag1_pix=(Hkk[:,0]+Hkk[:,1])/2
            V_diag2_pix=(Hkk[:,0]-Hkk[:,1])/2

            V_hori_pix = V_hori_pix/np.linalg.norm(V_hori_pix);
            V_vert_pix = V_vert_pix/np.linalg.norm(V_vert_pix);
            V_diag1_pix = V_diag1_pix/np.linalg.norm(V_diag1_pix);
            V_diag2_pix = V_diag2_pix/np.linalg.norm(V_diag2_pix);

            a1=V_hori_pix[0]
            b1=V_hori_pix[1]
            c1=V_hori_pix[2]

            a2=V_vert_pix[0]
            b2=V_vert_pix[1]
            c2=V_vert_pix[2]

            a3=V_diag1_pix[0]
            b3=V_diag1_pix[1]
            c3=V_diag1_pix[2]
            
            a4=V_diag2_pix[0]
            b4=V_diag2_pix[1]
            c4=V_diag2_pix[2]

            A_kk=np.zeros((2,2))
            A_kk[0,0]=a1*a2
            A_kk[0,1]=b1*b2
            A_kk[1,0]=a3*a4
            A_kk[1,1]=b3*b4
            b_kk=np.zeros((2,1))
            b_kk[0,0]=-(c1*c2)
            b_kk[1,0]=-(c3*c4)

            A_kk_row=A_kk.shape[0]
            A_kk_col=A_kk.shape[1]
            A=np.append(A,A_kk)
            A=A.reshape(A_kk_row*kk,A_kk_col)

            b_kk_row=b_kk.shape[0]
            b_kk_col=b_kk.shape[1]
            b=np.append(b,b_kk)
            b=b.reshape(b_kk_row*kk,b_kk_col)
            
    b=np.mat(b)
    A=np.mat(A)
    if two_focals_init==0:
        if ((b.T)*(sum(A.T).T))<0:
            two_focals_init = 1

    
    f_init=( np.sqrt( ( (b.T)*(sum(A.T).T) )/(b.T*b) ) )[0,0]*np.mat(np.ones((2,1)))

    alpha_init = 0

    KK=np.mat([[f_init[0,0],alpha_init*f_init[0,0],c_init[0,0]],[0,f_init[1,0],c_init[1,0]],[0,0,1]])
            
    inv_KK=inv(KK)

    cc=c_init.copy()
    fc=f_init.copy()
    kc=k_init.copy()
    alpha_c=copy.deepcopy(alpha_init)

    print '\n\nCalibration parameters after initialization:\n\n'
    print 'Focal Length:          fc = [ ',fc[0,0],' ',fc[1,0],' ]\n'
    print 'Principal point:       cc = [ ',cc[0,0],' ',cc[1,0],' ]\n'
    print 'Skew:             alpha_c = [ ',alpha_c,' ]   => angle of pixel = ',90 - math.atan(alpha_c)*180/math.pi,'degrees\n'
    print 'Distortion:            kc = [ ',kc[0,0],kc[1,0],kc[2,0],kc[3,0],kc[4,0],' ]\n'
    ###init_intrinsic_param end###

    ###go_calib_optim_iter###
    alpha_smooth=0.1
    est_fc=np.mat([[1.],[1.]])

    if np.prod(est_dist)==0:
        kc=np.mat(kc)
        kc=np.multiply(kc,est_dist)
    ###comp_ext_calib###
    N_points_views=np.mat(np.zeros((1,n_ima)))
    
    for kk in range(1,n_ima+1):
        exec('x_kk=x_'+str(kk)+'.copy()')
        exec('X_kk=X_'+str(kk)+'.copy()')
        N_points_views[0,kk-1] = x_kk.shape[1]
        ###comput_extrinsic function###
        extrinsic_init_out=compute_extrinsic_init(x_kk,X_kk,fc,cc,kc,alpha_c)
        omckk=extrinsic_init_out[0]
        Tckk=extrinsic_init_out[1]
        
        
        extrinsic_refine_out=compute_extrinsic_refine(omckk,Tckk,x_kk,X_kk,fc,cc,kc,alpha_c,20,thresh_cond)
        omckk=extrinsic_refine_out[0]
        Tckk=extrinsic_refine_out[1]
        Rckk=extrinsic_refine_out[2]
        JJ_kk=extrinsic_refine_out[3]

        
        #####################################
        if check_cond:
            if np.linalg.cond(JJ_kk)>thresh_cond:
                active_images[kk-1]=0
                omckk=0*np.mat(np.ones((3,1)))
                Tckk=0*np.mat(np.ones((3,1)))
                print '\nWarning: View #',kk,' ill-conditioned. This image is now set inactive.\n'
                desactivated_images.append(kk)

        
        exec('omc_'+str(kk)+'=omckk.copy()')
        exec('Tc_'+str(kk)+'=Tckk.copy()')
        
    ###comp_ext_calib end###

    ###go_calib_optim_iter###
    init_param=[fc,cc,alpha_c,kc,[0,0,0,0,0]]
    
    for kk in range(1,n_ima+1):
        exec('omckk=omc_'+str(kk)+'.copy()')
        exec('Tckk=Tc_'+str(kk)+'.copy()')
        init_param.append(omckk)
        init_param.append(Tckk)
        
        

    print '\nMain calibration optimization procedure - Number of images: ',len(ind_active),'\n'

    param=init_param[:]

    change=1
    Iter=0

    print 'Gradient descent iteration: '

    ###param to param_temp - type difference###
    param_size=np.size(param)

    param_temp=np.mat(np.zeros((15+(param_size-5)*3,1)))

    param_temp[0:2,0]=param[0]
    param_temp[2:4,0]=param[1]
    param_temp[4,0]=param[2]
    param_temp[5:10,0]=param[3]
    param_temp1=np.mat(param[4]).T
    param_temp[10:15,0]=param_temp1

    for kk in range(0,param_size-5):
        param_temp[kk*3+15:kk*3+18,0]=param[kk+5]
   
    param_list=param_temp.copy()
    

    
    while (change>(1e-9))&(Iter<MaxIter):
        print Iter+1,'...'

        f=param[0]
        c=param[1]
        alpha=param[2]
        k=param[3]
        
        N_points_views_active = N_points_views[0,ind_active.T]

        #JJ3=np.mat(np.zeros((15 + 6*n_ima,15 + 6*n_ima)))
        JJ3=sps.csr_matrix((15 + 6*n_ima,15 + 6*n_ima))
        ex3=np.mat(np.zeros((15+6*n_ima,1)))
        
        for kk in ind_active:
            omckk=param[5+2*kk]
            Tckk=param[6+2*kk]
            
           
            
            exec('x_kk=x_'+str(kk[0]+1)+'.copy()')
            exec('X_kk=X_'+str(kk[0]+1)+'.copy()')
        
            Np=N_points_views[0,kk[0]]
            
            
            project2_out = project_points2(X_kk,omckk,Tckk,f,c,k,alpha)

            x=project2_out[0]
            dxdom=project2_out[1]
            dxdT=project2_out[2]
            dxdf=project2_out[3]
            dxdc=project2_out[4]
            dxdk=project2_out[5]
            dxdalpha=project2_out[6]
            
            exkk = x_kk - x

            dxdf_row,dxdf_col=dxdf.shape
            dxdc_row,dxdc_col=dxdc.shape
            dxdalpha_row,dxdalpha_col=dxdalpha.shape
            dxdk_row,dxdk_col=dxdk.shape
            dxdom_row,dxdom_col=dxdom.shape
            dxdT_row,dxdT_col=dxdT.shape
            
            A=np.mat(np.zeros((dxdf_row,dxdf_col+dxdc_col+dxdalpha_col+dxdk_col)))
            A[:,0:dxdf_col]=dxdf
            A[:,dxdf_col:dxdf_col+dxdc_col]=dxdc
            A[:,dxdf_col+dxdc_col:dxdf_col+dxdc_col+dxdalpha_col]=dxdalpha
            A[:,dxdf_col+dxdc_col+dxdalpha_col:dxdf_col+dxdc_col+dxdalpha_col+dxdk_col]=dxdk
            
            A=A.T
            
            B=np.mat(np.zeros((dxdom_row,dxdom_col+dxdT_col)))

            B[:,0:dxdom_col]=dxdom
            B[:,dxdom_col:dxdom_col+dxdT_col]=dxdT

            B=B.T

            
            
            JJ3[0:10,0:10]=JJ3[0:10,0:10]+A*(A.T)
            JJ3[15+6*kk:15+6*kk + 6,15+6*kk:15+6*kk + 6]=B*(B.T)

            AB=A*(B.T)

            JJ3[0:10,15+6*kk:15+6*kk + 6] = AB
            JJ3[15+6*kk:15+6*kk + 6,0:10] = (AB).T

            ex3[0:10] = ex3[0:10] + A*np.reshape((exkk.T),(np.size(exkk),1))
            ex3[15+6*kk:15+6*kk + 6] = B*np.reshape((exkk.T),(np.size(exkk),1))

            if check_cond:
                JJ_kk=B.T
                if np.linalg.cond(JJ_kk)>thresh_cond:
                    active_images[kk]=0
                    print '\nWarning: View #',kk,' ill-conditioned. This image is now set inactive.(note: to disactivate this option, set check_cond=0)\n'
                    desactivated_images.append(kk)
                    param[5+2*kk]=np.mat(np.ones((3,1)))
                    param[6+2*kk]=np.mat(np.ones((3,1)))

        svs_temp1=center_optim*np.mat(np.ones((2,1)))
        
        svs_temp2=np.reshape((np.mat(np.ones((6,1)))*np.mat(active_images)).T,(6*n_ima,1))
        est_fc_row,est_fc_col=est_fc.shape
        est_dist_row,est_dist_col=est_dist.shape
        svs_temp1_row,svs_temp1_col=svs_temp1.shape
        svs_temp2_row,svs_temp2_col=svs_temp2.shape
        selected_variables=np.mat(np.zeros((est_fc_row+svs_temp1_row+6+est_dist_row+svs_temp2_row,1)))
        selected_variables[0:est_fc_row,0]=est_fc
        selected_variables[est_fc_row:est_fc_row+svs_temp1_row,0]=svs_temp1
        selected_variables[est_fc_row+svs_temp1_row,0]=est_alpha
        selected_variables[est_fc_row+svs_temp1_row+1:est_fc_row+svs_temp1_row+1+est_dist_row,0]=est_dist
        selected_variables[est_fc_row+svs_temp1_row+6+est_dist_row:est_fc_row+svs_temp1_row+6+est_dist_row+svs_temp2_row,0]=svs_temp2
        
        ind_Jac=np.where(selected_variables)[0]
       
        rep_ind_Jac=numpy.matlib.repmat(ind_Jac,ind_Jac.shape[1],1)
        JJ3=JJ3[rep_ind_Jac.T,rep_ind_Jac]
        
        ex3=ex3[ind_Jac,0].T
        
        
        JJ2_inv=sps.csr_matrix(inv((JJ3.todense()).T))

        alpha_smooth2 = 1-(1-alpha_smooth)**(Iter+1)

        param_innov = alpha_smooth2*JJ2_inv*ex3


        param_size=np.size(param)

        param_temp=np.mat(np.zeros((15+(param_size-5)*3,1)))

        param_temp[0:2,0]=param[0]
        param_temp[2:4,0]=param[1]
        param_temp[4,0]=param[2]
        param_temp[5:10,0]=param[3]
        param_temp1=np.mat(param[4]).T
        param_temp[10:15,0]=param_temp1

        for kk in range(0,param_size-5):
            param_temp[kk*3+15:kk*3+18,0]=param[kk+5]
        
        param_up = param_temp[ind_Jac,0].T + param_innov

        param_temp[ind_Jac,0] = param_up.T
        
        fc_current = param_temp[0:2]
        cc_current = param_temp[2:4]

        if center_optim:
            if ((param_temp[2]<0)|(param_temp[2]>nx)|(param_temp[3]<0)|(param_temp[3]>ny)):
                print 'Warning: it appears that the principal point cannot be estimated. Setting center_optim = 0\n'
                center_optim=0
                cc_current= c
            else:
                cc_current = param_temp[2:4]
                
        else:
            cc_current = param_temp[2:4]

        alpha_current = param_temp[4,0]
        kc_current =param_temp[5:10]

        if (est_aspect_ratio==0) & (est_fc.all()==np.mat([[1],[1]]).all()):
            fc_current[1] = fc_current[0]
            param_temp[1] = param_temp[0]

        fc_current_row,fc_current_col=fc_current.shape
        cc_current_row,cc_current_col=cc_current.shape

        fccc=np.mat(np.zeros((fc_current_row+cc_current_row,fc_current_col)))
        fccc[0:fc_current_row,:]=fc_current
        fccc[fc_current_row:fc_current_row+cc_current_row,:]=cc_current

        f_row,f_col=f.shape
        c_row,c_col=c.shape
        fc_temp=np.mat(np.zeros((f_row+c_row,f_col)))
        fc_temp[0:f_row,:]=f
        fc_temp[f_row:f_row+c_row,:]=c

        change = np.linalg.norm(fccc-fc_temp)/np.linalg.norm(fccc)

        
        
        if recompute_extrinsic:
            MaxIter2=20
            for kk in ind_active:
                omc_current = param_temp[15+6*(kk[0]):15+6*(kk[0]) + 3]
                Tc_current = param_temp[15+6*(kk[0])+3:15+6*(kk[0]) + 6]
                exec('X_kk=X_'+str(kk[0]+1)+'.copy()')
                exec('x_kk=x_'+str(kk[0]+1)+'.copy()')
                extrinsic_init_out=compute_extrinsic_init(x_kk,X_kk,fc_current,cc_current,kc_current,alpha_current)
                omc_current=extrinsic_init_out[0]
                Tc_current=extrinsic_init_out[1]
                
                
                extrinsic_refine_out=compute_extrinsic_refine(omc_current,Tc_current,x_kk,X_kk,fc_current,cc_current,kc_current,alpha_current,MaxIter2,thresh_cond)
                omckk=extrinsic_refine_out[0]
                Tckk=extrinsic_refine_out[1]
                Rckk=extrinsic_refine_out[2]
                JJ_kk=extrinsic_refine_out[3]
                
                if check_cond:
                    if (np.linalg.cond(JJ_kk)>thresh_cond):
                        active_images[kk[0]]=0
                        print ('\nWarning: View #%d ill-conditioned. This image is now set inactive. (note: to disactivate this option, set check_cond=0)\n')
                        desactivated_images=desactivated_images.append(kk[0]+1)
                        omckk=None
                        Tckk=None

                param_temp[15+6*(kk[0]):15+6*(kk[0]) + 3]=omckk
                param_temp[15+6*(kk[0])+3:15+6*(kk[0]) + 6]=Tckk

        ####param_temp to param - dont use now###
        param[0]=param_temp[0:2,0]
        param[1]=param_temp[2:4,0]
        param[2]=param_temp[4,0]
        param[3]=param_temp[5:10,0]
        param[4][0]=param_temp[10,0]
        param[4][1]=param_temp[11,0]
        param[4][2]=param_temp[12,0]
        param[4][3]=param_temp[13,0]
        param[4][4]=param_temp[14,0]

        for kk in range(0,param_size-5):
            param[5+kk]=param_temp[kk*3+15:kk*3+18,0]

        ###########################################


            
        param_list_row,param_list_col=param_list.shape
        param_list_temp=np.mat(np.zeros((param_list_row,param_list_col+1)))
        param_list_temp[:,0:param_list_col]=param_list
        param_list_temp[:,param_list_col]=param_temp
        param_list=param_list_temp


        Iter=Iter+1
        
    
    print 'done\n'
    print 'Estimation of uncertainties...'
    
    solution = param_temp
    
    fc=solution[0:2,0]
    cc=solution[2:4,0]
    alpha_c=solution[4,0]
    kc=solution[5:10,0]

    for kk in range(0,n_ima):
        if active_images[kk]:
            omckk=solution[15+6*(kk):15+6*(kk) + 3,0]
            Tckk=solution[15+6*(kk) + 3:15+6*(kk) + 6,0]
            rodri_out=rodrigues(omckk)
            Rckk=rodri_out[0]
        else:
            omckk=None
            Tckk=None
            Rckk=None
        exec('omc_'+str(kk+1)+'=omckk.copy()')
        exec('Rc_'+str(kk+1)+'=Rckk.copy()')
        exec('Tc_'+str(kk+1)+'=Tckk.copy()')



    ### comp_error_calib ### 
    ex=np.mat(np.zeros((2,1)))
    x=np.mat(np.zeros((2,1)))
    y=np.mat(np.zeros((2,1)))

    for kk in range(0,n_ima):
        exec('omckk=omc_'+str(kk+1)+'.copy()')
        exec('Tckk=Tc_'+str(kk+1)+'.copy()')

        if (active_images[kk])&(('omckk' in globals().keys())|('omckk' in locals().keys())):
            exec('project2_out=project_points2(X_'+str(kk+1)+',omckk,Tckk,fc,cc,kc,alpha_c)')
            exec('y_'+str(kk+1)+'=project2_out[0]')
            exec('ex_'+str(kk+1)+'=x_'+str(kk+1)+'-y_'+str(kk+1))
            exec('x_kk=x_'+str(kk+1)+'.copy()')
            if ex.shape[1]!=1:
                ex_row,ex_col=ex.shape
                exec('ex_'+str(kk+1)+'_row,ex_'+str(kk+1)+'_col=ex_'+str(kk+1)+'.shape')
                exec('ex_temp=np.mat(np.zeros((ex_row,ex_col+ex_'+str(kk+1)+'_col)))')
                exec('ex_temp[:,0:ex_col]=ex')
                exec('ex_temp[:,ex_col:ex_col+ex_'+str(kk+1)+'_col]=ex_'+str(kk+1))
                ex=ex_temp.copy()

                x_row,x_col=x.shape
                exec('x_'+str(kk+1)+'_row,x_'+str(kk+1)+'_col=x_'+str(kk+1)+'.shape')
                exec('x_temp=np.mat(np.zeros((x_row,x_col+x_'+str(kk+1)+'_col)))')
                exec('x_temp[:,0:x_col]=x')
                exec('x_temp[:,x_col:x_col+x_'+str(kk+1)+'_col]=x_'+str(kk+1))
                x=x_temp.copy()

                y_row,y_col=y.shape
                exec('y_'+str(kk+1)+'_row,y_'+str(kk+1)+'_col=y_'+str(kk+1)+'.shape')
                exec('y_temp=np.mat(np.zeros((y_row,y_col+y_'+str(kk+1)+'_col)))')
                exec('y_temp[:,0:y_col]=y')
                exec('y_temp[:,y_col:y_col+y_'+str(kk+1)+'_col]=y_'+str(kk+1))
                y=y_temp.copy()
            else:
                exec('ex=ex_'+str(kk+1)+'.copy()')
                exec('x=x_'+str(kk+1)+'.copy()')
                exec('y=y_'+str(kk+1)+'.copy()')                            
        else:
            exec('ex_'+str(kk+1)+'=None')

    err_std=np.std(ex,axis=1)
    ex_temp=np.reshape(ex.T,(np.size(ex),1))
    sigma_x=np.std(ex_temp,axis=0)
    N_points_views_active = N_points_views[0,ind_active.T]
    JJ3=sps.csr_matrix((15 + 6*n_ima,15 + 6*n_ima))
    for kk in ind_active:
        omckk = param_temp[15+6*(kk[0]):15+6*(kk[0]) + 3,0]
        Tckk = param_temp[15+6*(kk[0])+3:15+6*(kk[0]) + 6,0]
        exec('X_kk=X_'+str(kk[0]+1)+'.copy()')
        Np=N_points_views[0,kk[0]]

        project2_out = project_points2(X_kk,omckk,Tckk,fc,cc,kc,alpha_c)

        x=project2_out[0]
        dxdom=project2_out[1]
        dxdT=project2_out[2]
        dxdf=project2_out[3]
        dxdc=project2_out[4]
        dxdk=project2_out[5]
        dxdalpha=project2_out[6]

        A=np.mat(np.zeros((dxdf_row,dxdf_col+dxdc_col+dxdalpha_col+dxdk_col)))
        A[:,0:dxdf_col]=dxdf
        A[:,dxdf_col:dxdf_col+dxdc_col]=dxdc
        A[:,dxdf_col+dxdc_col:dxdf_col+dxdc_col+dxdalpha_col]=dxdalpha
        A[:,dxdf_col+dxdc_col+dxdalpha_col:dxdf_col+dxdc_col+dxdalpha_col+dxdk_col]=dxdk
        
        A=A.T
        
        B=np.mat(np.zeros((dxdom_row,dxdom_col+dxdT_col)))

        B[:,0:dxdom_col]=dxdom
        B[:,dxdom_col:dxdom_col+dxdT_col]=dxdT

        B=B.T

        
        
        JJ3[0:10,0:10]=JJ3[0:10,0:10]+A*(A.T)
        JJ3[15+6*kk:15+6*kk + 6,15+6*kk:15+6*kk + 6]=B*(B.T)

        AB=A*(B.T)

        JJ3[0:10,15+6*kk:15+6*kk + 6] = AB
        JJ3[15+6*kk:15+6*kk + 6,0:10] = (AB).T
        
    
    JJ3=JJ3[rep_ind_Jac.T,rep_ind_Jac]
    JJ2_inv=sps.csr_matrix(inv((JJ3.todense()).T))

    param_error=np.mat(np.zeros((6*n_ima+15,1)))

    param_error[ind_Jac,0] =  (3*np.sqrt( np.mat(np.diag(JJ2_inv.todense())).T )*sigma_x[0,0]).T
    
    solution_error = param_error.copy()      

    fc=solution[0:2]
    cc=solution[2:4]
    alpha_c=solution[4]
    kc=solution[5:10]

    fc_error = solution_error[0:2]
    cc_error = solution_error[2:4]
    alpha_c_error = solution_error[4]
    kc_error = solution_error[5:10]

    
    KK=np.mat(np.zeros((3,3)))
    KK[0,0]=fc[0,0]
    KK[0,1]=fc[0,0]*alpha_c
    KK[0,2]=cc[0,0]
    KK[1,1]=fc[1,0]
    KK[1,2]=cc[1,0]
    KK[2,2]=1

    inv_KK=inv(KK)
    
    
    for kk in range(0,n_ima):
        if active_images[kk]:
            omckk = solution[15+6*(kk):15+6*(kk) + 3]
            Tckk = solution[15+6*(kk) + 3:15+6*(kk) + 6]
            omckk_error = solution_error[15+6*(kk):15+6*(kk) + 3]
            Tckk_error = solution_error[15+6*(kk) + 3:15+6*(kk) + 6]
          
            rodri_out = rodrigues(omckk)
            Rckk=rodri_out[0]
            Rckk_temp=np.mat(np.zeros((3,3)))
            Rckk_temp[:,0]=Rckk[:,0]
            Rckk_temp[:,1]=Rckk[:,1]
            Rckk_temp[:,2]=Tckk
            Hkk = KK * Rckk_temp
            Hkk = Hkk / Hkk[2,2]
        else:
            omckk = None
            Tckk = None
            Rckk = None
            Hkk = None
            omckk_error =None
            Tckk_error = None
          
       
               
        exec ('omc_'+str(kk)+' = omckk.copy()')
        exec ('Rc_'+str(kk)+' = Rckk.copy()')
        exec ('Tc_'+str(kk)+' = Tckk.copy()')
        exec ('H_'+str(kk)+'= Hkk.copy()')
        exec ('omc_error_'+str(kk)+' = omckk_error.copy()')
        exec ('Tc_error_'+str(kk)+' = Tckk_error.copy()')
       
        
    print 'done\n'
    print '\n\nCalibration results after optimization (with uncertainties):\n\n'
    print 'Focal Length:          fc = [ ',fc[0],fc[1],']?[',fc_error[0],fc_error[1],']\n'
    print 'Principal point:       cc = [ ',cc[0],cc[1],']?[',cc_error[0],cc_error[1],']\n'
    print 'Skew:             alpha_c = [ ',alpha_c,']?[',alpha_c_error,'=> angle of pixel axes =',90 - np.arctan(alpha_c)*180/math.pi ,'?',np.arctan(alpha_c_error)*180/math.pi,'degrees\n'
    print 'Distortion:            kc = [ ',kc[0],kc[1],kc[2],kc[3],kc[4],'] ?[', kc_error[0],kc_error[1],kc_error[2],kc_error[3],kc_error[4],' ]\n'   
    print 'Pixel error:          err = [ ',err_std[0],err_std[1],' ]\n\n' 
    print 'Note: The numerical errors are approximately three times the standard deviations (for reference).\n\n\n'


    print '\n\nCalibration results R,T'
    for kk in range(0,n_ima):
        exec('Tckk=Tc_'+str(kk+1)+'.copy()')
        exec('Rckk=Rc_'+str(kk+1)+'.copy()')
        print 'T',kk+1,' : [',Tckk[0], Tckk[0], Tckk[0],']\n'
        print 'R',kk+1,' : [',Rckk[0,0], Rckk[0,1], Rckk[0,2],']\n'
        print '\t[',Rckk[1,0], Rckk[1,1], Rckk[1,2],']\n'
        print '\t[',Rckk[2,0], Rckk[2,1], Rckk[2,2],']\n'
        

    
#########compute_homography###################
def compute_homography(m,M):
    global kbw
    Np=m.shape[1]
    if m.shape[0]<3:
        mm=np.ones((m.shape[0]+1,m.shape[1]))
        mm[0:m.shape[0],:]=m
        m=mm.copy()
    if M.shape[0]<3:
        MM=np.ones((M.shape[0]+1,M.shape[1]))
        MM[0:M.shape[0],:]=M
        M=MM.copy()
    m=np.multiply(m,1/(np.ones((3,1))*m[2,:]))
    M=np.multiply(M,1/(np.ones((3,1))*M[2,:]))
    
    ax = m[0,:]
    ay = m[1,:]

    mxx = np.mean(ax)
    myy = np.mean(ay)
    ax = ax-mxx
    ay = ay-myy

    scxx = np.mean(abs(ax))
    scyy = np.mean(abs(ay))
    Hnorm = np.mat([[1/scxx,0,-mxx/scxx],[0,1/scyy,-myy/scyy],[0,0,1]])
    inv_Hnorm = np.mat([[scxx,0,mxx],[0,scyy,myy],[0,0,1]])

    mn = Hnorm*m

    L=np.zeros((2*Np,9))
    L[0:2*Np:2,0:3]=M.T
    L[1:2*Np:2,3:6]=M.T
    L[0:2*Np:2,6:9]=-(np.multiply(np.ones((3,1))*mn[0,:],M)).T
    L[1:2*Np:2,6:9]=-(np.multiply(np.ones((3,1))*mn[1,:],M)).T
    L=np.mat(L)
    if Np>4:
        L=L.T*L

    U,S,V=np.linalg.svd(L)
    V=V.T
    hh=V[:,8]
    hh=hh/hh[8]
    Hrem=np.reshape(hh,(3,3))
    H = inv_Hnorm*Hrem
    H=np.mat(H)
    if Np>4:
        hhv=H.reshape(9,1)
        hhv=hhv[0:8,0]

        for hiter in range(1,11):
            
            mrep=H*M
            J=np.mat(np.zeros((2*Np,8)))
            MMM=np.multiply(M,1/(np.mat(np.ones((3,1)))*mrep[2,:]))

            J[0:2*Np:2,0:3]=-(MMM.T)
            J[1:2*Np:2,3:6]=-(MMM.T)

            mrep=np.multiply(mrep,1/(np.mat(np.ones((3,1)))*mrep[2,:]))

            m_err = m[0:2,:]-mrep[0:2,:]
            m_err_row=np.size(m_err)
            m_err = (m_err.T).reshape(m_err_row,1)

            MMM2=np.multiply(np.mat(np.ones((3,1)))*mrep[0,:],MMM)
            MMM3=np.multiply(np.mat(np.ones((3,1)))*mrep[1,:],MMM)

            J[0:2*Np:2,6:8]=MMM2[0:2,:].T
            J[1:2*Np:2,6:8]=MMM3[0:2,:].T
            
            MMM=np.multiply(M,1/(np.mat(np.ones((3,1)))*mrep[2,:])).T

            hh_innov=inv((J.T)*J)*(J.T)*m_err
            hhv_up=np.mat(np.ones((hhv.shape[0]+1,hhv.shape[1])))
            
            hhv_up[0:hhv.shape[0],:] = hhv - hh_innov
            
            H_up = hhv_up.reshape(3,3)
            
            hhv=hhv_up[0:8,:]
        H=H_up.copy()

        
    return H,Hnorm,inv_Hnorm
#############count_squares##################
def count_squares(I,x1,y1,x2,y2,win):
    global kbw1
    ny=np.size(I,0)
    nx=np.size(I,1)
    if (x1-win <= 0) | (x1+win >= nx) | (y1-win <= 0) | (y1+win >= ny) |(x2-win <= 0) | (x2+win >= nx) | (y2-win <= 0) | (y2+win >= ny):
        ns = -1
        return ns
    if ((x1-x2)**2+(y1-y2)**2)<win:
        ns = -1
        return ns
    llambda = np.mat([[y1[0,0]-y2[0,0]],[x2[0,0]-x1[0,0]],[x1[0,0]*y2[0,0]-x2[0,0]*y1[0,0]]])
    llambda=1/np.sqrt(llambda[0,0]**2+llambda[1,0]**2)*llambda
    l1=llambda + np.mat([[0.],[0.],[win]])
    l2=llambda - np.mat([[0.],[0.],[win]])
    dx = x2-x1
    dy = y2 - y1
    if abs(dx)>abs(dy):
        if x2[0,0]>x1[0,0]:
            xs=np.arange(x1[0,0],x2[0,0]+0.00000001)
        else :
            xs=np.arange(x1[0,0],x2[0,0]-0.00000001,-1)
        ys = -(llambda[2,0]+llambda[0,0]*xs)/llambda[1,0]
    else :
        if y2[0,0]>y1[0,0]:
            ys=np.arange(y1[0,0],y2[0,0]+0.00000001)
        else:
            ys=np.arange(y1[0,0],y2[0,0]-0.00000001,-1)
        xs = -(llambda[2,0]+llambda[1,0]*ys)/llambda[0,0]
    Np=len(xs)
    xs_mat=np.ones((2*win+1,1))*xs
    ys_mat=np.ones((2*win+1,1))*ys    
    win_mat=(np.mat(np.arange(-win,win+0.00001)).T)*np.ones((1,Np))
    xs_mat2=np.round(xs_mat-win_mat*llambda[0,0])
    ys_mat2=np.round(ys_mat-win_mat*llambda[1,0])
    ind_mat = (xs_mat2-1)*ny+ys_mat2
    ima_patch=np.zeros((2*win+1,Np))
    indshape=ind_mat.shape

    for j in range(indshape[1]):
        for i in range(indshape[0]):
            ima_patch[i,j]=I[(ind_mat[i,j]-1)%480][math.floor((ind_mat[i,j]-1)/480)]

    filtk=np.ones((win*2+1,Np))
    filtk[win,:]=np.zeros((1,Np))
    filtk[(win+1):,:]=-np.ones((win,Np))
    filtk=np.mat(filtk)
    out_f=sum(np.multiply(filtk,ima_patch))
    out_f_f=signal.convolve2d(out_f,np.mat([1./4,1./2,1./4]),'same')
    out_f_f=out_f_f[0,win:out_f_f.shape[1]-win]
    ns_temp=(out_f_f[1:out_f_f.shape[0]]>=0)&(out_f_f[0:out_f_f.shape[0]-1]<0)|((out_f_f[1:out_f_f.shape[0]]<=0)&(out_f_f[0:out_f_f.shape[0]-1]>0))
    ns_ind=[]
    for i in range(ns_temp.shape[0]):
        if ns_temp[i]==True:
            ns_ind.append(i)
    ns=len(ns_ind)+1
    return ns

#############cornerfinder########
def cornerfinder(xt,I,wintx,winty):
    global kbw
    global kbw1
    line_feat =1
    xt=xt.T
    xt=np.fliplr(xt)
    
    wx2=-1
    wy2=-1    
    mask = np.exp(-np.mat(((arange(-wintx,wintx+1,dtype=float))/wintx)**2).T) * np.exp(-np.mat((arange(-winty,winty+1,dtype=float)/winty)**2))
    X,Y=np.meshgrid(arange(-wintx,wintx+1,dtype=float),arange(-winty,winty+1,dtype=float))
    mask2 = X**2+Y**2
    mask2[wintx,winty]=1
    mask2=1/mask2
    ###no problem both offx, offy are float
    offx=(np.mat(arange(-wintx,wintx+1,dtype=float)).T)*np.ones([1,2*winty+1],dtype=float)
    offy=np.ones([2*wintx+1,1])*arange(-winty,winty+1)

    resolution = 0.005
    MaxIter = 10
    nx = np.size(I,0)
    ny = np.size(I,1)
    
    N=np.size(xt,0)
    
    xc=xt.copy()
    ttype = np.zeros([1,N]) 
    
    for i in range(0,N):    #0~N-1
        v_extra = resolution +1
        compt =0
        while (np.linalg.norm(v_extra)>resolution) & (compt<MaxIter):
            cIx=xc[i,0]
            cIy=xc[i,1]
            crIx=round(cIx)
            crIy=round(cIy)
            itIx=cIx-crIx
            itIy=cIy-crIy
            if itIx > 0:
                vIx = (np.mat([itIx, 1-itIx, 0])).T
            else:
                vIx = (np.mat([0, 1+itIx, -itIx])).T

            if itIy > 0:
                vIy = (np.mat([itIy, 1-itIy, 0]))
            else:
                vIy = (np.mat([0, 1+itIy, -itIy]))

            if crIx-wintx-2 < 0 :
                xmin=0
                xmax = 2*wintx+4
            elif crIx+wintx+2 > nx-1:
                xmax = nx-1
                xmin = nx-2*wintx-5
            else:
                xmin = crIx-wintx-2
                xmax = crIx+wintx+2
            if crIy-winty-2 < 0:
                ymin=0
                ymax = 2*winty+4
            elif crIy+winty+2 > ny-1:
                ymax = ny-1
                ymin = ny-2*winty-5
            else:
                ymin = crIy-winty-2
                ymax = crIy+winty+2
            
            SI = I[xmin:xmax+1,ymin:ymax+1]
            SI = signal.convolve2d(signal.convolve2d(SI,vIx,'same'),vIy,'same')
            SI = SI[1:2*wintx+4,1:2*winty+4]
            
            gx,gy = np.gradient(SI)
            gx = gx[1:2*wintx+2,1:2*winty+2]
            gy = gy[1:2*wintx+2,1:2*winty+2]

            px=cIx+offx
            py=cIy+offy

            gxx = np.multiply(np.multiply(gx,gx),mask)
            gyy = np.multiply(np.multiply(gy,gy),mask)
            gxy = np.multiply(np.multiply(gx,gy),mask)
            
            bb=np.mat([np.sum(np.multiply(gxx,px)+np.multiply(gxy,py)),np.sum(np.multiply(gxy,px)+np.multiply(gyy,py))]).T

            a=np.sum(gxx)
            b=np.sum(gxy)
            c=np.sum(gyy)
            
            dt=a*c-(b**2)
            
            xc2 = np.mat([float(c*bb[0]-b*bb[1]), a*bb[1]-b*bb[0]])/dt
            
            
            if line_feat:
                G=np.mat([[a,b],[b,c]])
                U,S,V=np.linalg.svd(G)
                
                if S[0]/S[1]>50:
                    xc2=xc2 + np.sum( np.multiply( (xc[i,:]-xc2) , ((V[:,1]).T) ) )*(V[:,1].T)
                    ttype[0,i]=1
            
            v_extra = xc[i,:]-xc2
            xc[i,:] = xc2
            compt = compt +1
            
            
    delta_x = xc[:,0]-xt[:,0]
    delta_y = xc[:,1]-xt[:,1]
    
    bad = (abs(delta_x)>wintx)|(abs(delta_y)>winty)
    good = ~bad
    in_bad = np.where(bad>0)
    xc[in_bad,:]=xt[in_bad,:]
            
    xc=np.fliplr(xc)
    xc=xc.T
    bad=bad.T
    good=good.T
            
    return xc, good, bad, ttype             
            
####################click calib#######################
def click_calib():
    global ima_numbers
    global ima_proc
    global kk_first
    global wintx
    global winty
    global manual_squares
    global x
    global y
    global point
    global n_sq_x_default
    global n_sq_y_default
    global dX_default
    global dY_default
    global kbw
    global kbw1
    active_images=[]
       
    ima_numbers = [input('Number(s) of image(s) to process (-1 = all images) = ')]
    print ima_numbers   #1~20
    if ima_numbers[0]<0:
        ima_proc.append(range(1,n_ima+1))
        kk_first = ima_proc[0]
    else:
        ima_proc = [ima_numbers]
        kk_first=ima_proc
    print '\n',ima_proc
    print 'Window size for corner finder (wintx and winty):' 
    wintx = input('wintx('+str(wintx_default)+')=')
    winty = input('winty('+str(winty_default)+')=')
    wintx=round(wintx)
    winty=round(winty)
    #xsquare=np.mat([wintx+.5, -(wintx+.5), -(wintx+.5), wintx+.5, wintx+.5])
    #ysquare=np.mat([winty+.5, winty+.5, -(winty+.5), -(winty+.5), winty+.5])
    print 'Window size =',2*wintx+1,'x',2*winty+1,'\n'
    print 'Do you want to use the automatic square counting mechanism (0=default)'
    manual_squares = input(' or do you always want to enter the number of squares manually (1,other)? ')
    print manual_squares
    
    for kk in ima_proc[0]:
        print '\nProcessing image', kk,'...\n'
        print 'Using (wintx, winty)=(', wintx, winty,') - Window size =', 2*wintx+1,'x',2*winty+1
        exec('Ii=I_'+str(kk))
        print 'Click on the four extreme corners of the rectangular complete pattern (the first clicked corner is the origin)...'
        x=[]
        y=[]        
        imagefigure=plt.figure(kk)
        figuremanager=plt.get_current_fig_manager()
        figuremanager.window.geometry("+700+100")
        plt.imshow(Ii,cmap=cm.Greys_r)
        plt.axis([0,nx,ny,0])
        plt.title('Click on the four corners... Image'+str(kk))
        for count in range(0,4):
            point=plt.ginput(1)
            point=np.mat(point)
            
            cornerout=cornerfinder(point.T,Ii,winty,wintx)
            xxi=np.mat(cornerout[0])
                        
            xi = xxi[0,0]
            yi = xxi[1,0]
            
            x.append(xi)
            y.append(yi)
           
            clickcornerplot=plt.plot(x,y,'+',color=[1.,0.314,0.51],markersize=10,markeredgewidth=3)
            xmat=np.mat(xi)
            ymat=np.mat(yi)
            #kbw=xmat+np.mat([wintx+.5, -(wintx+.5), -(wintx+.5), wintx+.5, wintx+.5])
            #plt.plot(xmat+xsquare, ymat+ysquare, color=[1.,0.314,0.51],linewidth=10)
            clicklineplot=plt.plot(x,y, linewidth=2.0, color=[1, 0, 0.7])
        Xii=x[:]
        Yii=y[:]
        x.append(x[0])
        y.append(y[0])
        clicklineplot=plt.plot(x,y, linewidth=2.0, color=[1, 0, 0.7])
        plt.draw()
        #print np.mat([Xii,Yii]).T
        cornerout2=cornerfinder(np.mat([Xii,Yii]),Ii,winty,wintx)
        
        x=cornerout2[0][0,:].T
        y=cornerout2[0][1,:].T
        x_mean=np.mean(x)
        y_mean=np.mean(y)
        x_v=x-x_mean
        y_v=y-y_mean
        theta=np.arctan2(-y_v,x_v)
        sorttheta=(theta-theta[0])%(2*math.pi)
        ind=sorted(range(len(sorttheta)),key=sorttheta.__getitem__)
        ind=[item for item in reversed(ind)]
        x1=x[ind[0]]
        x2=x[ind[1]]
        x3=x[ind[2]]
        x4=x[ind[3]]
        
        y1=y[ind[0]]
        y2=y[ind[1]]
        y3=y[ind[2]]
        y4=y[ind[3]]
        x=np.mat([[x1[0,0]],[x2[0,0]],[x3[0,0]],[x4[0,0]]])
        y=np.mat([[y1[0,0]],[y2[0,0]],[y3[0,0]],[y4[0,0]]])
        
        p_center = np.cross(np.cross(np.array([x1[0,0],y1[0,0],1]),np.mat([x3[0,0],y3[0,0],1])),np.cross(np.array([x2[0,0],y2[0,0],1]),np.mat([x4[0,0],y4[0,0],1])))
        
        x5=p_center[0,0]/p_center[0,2]
        y5=p_center[0,1]/p_center[0,2]
        x6 = (x3 + x4)/2
        y6 = (y3 + y4)/2
        x7 = (x1 + x4)/2
        y7 = (y1 + y4)/2
        vX=np.mat([[x6[0,0]-x5],[y6[0,0]-y5]])
        vX = vX/np.linalg.norm(vX)
        vY=np.mat([[x7[0,0]-x5],[y7[0,0]-y5]])
        vY = vY/np.linalg.norm(vY)
        vO=np.mat([[x4[0,0]-x5],[y4[0,0]-y5]])
        vO = vO/np.linalg.norm(vO)

        delta = 30
        
        plt.close(imagefigure)
        imagefigure=plt.figure(kk+kk*10)
        figuremanager=plt.get_current_fig_manager()
        figuremanager.window.geometry("+700+100")
        plt.imshow(Ii,cmap=cm.Greys_r)
        plt.axis([0,nx,ny,0])
        plt.title('the four corners... Image'+str(kk))
        plt.plot(np.append([x[:,0]],[x[0,0]]),np.append([y[:,0]],[y[0,0]]),'g-')
        plt.plot(x,y,'og')
        hx=plt.text(x6 + delta * vX[0,0] ,y6 + delta*vX[1,0],'X',color='g',size=14)
        hy=plt.text(x7 + delta * vY[0,0] ,y7 + delta*vY[1,0],'Y',color='g',size=14)
        hy=plt.text(x4 + delta * vO[0,0] ,y4 + delta*vO[1,0],'O',color='g',size=14)
        plt.draw()
        plt.pause(.1)

        if manual_squares:
            n_sq_x=input(['Number of squares along the X direction ([]='+str(n_sq_x_default)+')='])
            n_sq_x=input(['Number of squares along the X direction ([]='+str(n_sq_x_default)+')='])
        else :
            n_sq_x1 = count_squares(Ii,x1,y1,x2,y2,wintx)
            n_sq_x2 = count_squares(Ii,x3,y3,x4,y4,wintx)
            n_sq_y1 = count_squares(Ii,x2,y2,x3,y3,wintx)
            n_sq_y2 = count_squares(Ii,x4,y4,x1,y1,wintx)
            n_sq_x = n_sq_x1
            n_sq_y = n_sq_y1
            
        n_sq_x_default=n_sq_x
        n_sq_y_default-n_sq_y
        if (kk==1)|(np.size(ima_proc)==1):            
            dX=input(['Size dX of each square along the X direction ([]='+str(dX_default)+'mm)='])
            dY=input(['Size dY of each square along the Y direction ([]='+str(dY_default)+'mm)='])
        if (dX!=dX_default)|(dY!=dY_default):
            dX=input(['Type again size dX of each square along the X direction ([]='+str(dX_default)+'mm)='])
            dY=input(['Type again Size dY of each square along the Y direction ([]='+str(dY_default)+'mm)='])
            dX_default=dX
            dY_default=dY
        
            

        a00=np.mat([[x[0,0]],[y[0,0]],[1]])
        a10=np.mat([[x[1,0]],[y[1,0]],[1]])
        a11=np.mat([[x[2,0]],[y[2,0]],[1]])
        a01=np.mat([[x[3,0]],[y[3,0]],[1]])
        a001=np.mat([[x[0,0],x[1,0],x[2,0],x[3,0]],[y[0,0],y[1,0],y[2,0],y[3,0]],[1,1,1,1]])
        a002=np.mat([[0, 1, 1, 0],[0, 0, 1, 1],[1, 1, 1, 1]])
                
        homographyout=compute_homography(a001,a002)
        
        Homo=homographyout[0]
        Hnorm=homographyout[1]
        inv_Hnorm=homographyout[2]
        x_l=(np.mat(range(n_sq_x+1)).T*np.ones((1,n_sq_y+1)))/n_sq_x
        y_l=(np.ones((n_sq_x+1,1))*np.mat(range(n_sq_y+1)))/n_sq_y
        pts=np.ones((3,(n_sq_x+1)*(n_sq_y+1)))
        for j in range(x_l.shape[1]):
            for i in range(x_l.shape[0]):
                pts[0,i+j*x_l.shape[0]]=x_l[i,j]
                pts[1,i+j*x_l.shape[0]]=y_l[i,j]
        pts=np.mat(pts)
        XX=Homo*pts
        XX=np.multiply(XX[0:2,:],1/(np.mat(np.ones((2,1)))*XX[2,:]))
        W = n_sq_x*dX
        L = n_sq_y*dY
        Np = (n_sq_x+1)*(n_sq_y+1)
        print 'Corner extraction...'
        cornerout3 = cornerfinder(XX,Ii,winty,wintx)
        grid_pts=cornerout3[0]
        grid_pts = grid_pts - 1

        
        ind_corners=np.mat([0, n_sq_x, ((n_sq_x+1)*n_sq_y),(n_sq_x+1)*(n_sq_y+1)-1])
        ind_orig=((n_sq_x+1)*n_sq_y)
        
        xorig=grid_pts[0,ind_orig]
        yorig=grid_pts[1,ind_orig]
        dxpos=np.zeros((2,grid_pts.shape[0]))
        for i in range(grid_pts.shape[0]):
            dxpos[0,i]=np.mat(grid_pts[i,ind_orig])
            dxpos[1,i]=np.mat(grid_pts[i,ind_orig+1])
        dxpos=np.mat(dxpos)
        dxpos=np.mean(dxpos,axis=0)
        dypos=np.zeros((2,grid_pts.shape[0]))
        for i in range(grid_pts.shape[0]):
            dypos[0,i]=np.mat(grid_pts[i,ind_orig])
            dypos[1,i]=np.mat(grid_pts[i,ind_orig-n_sq_x-1])
        dypos=np.mat(dypos)
        dypos=np.mean(dypos,axis=0)
        
        raw_input('Press enter to continue: ')
        plt.close(imagefigure)
        
        imagefigure2=plt.figure(kk*100)
        figuremanager=plt.get_current_fig_manager()
        figuremanager.window.geometry("+700+100")
        plt.imshow(Ii,cmap=cm.Greys_r)
        plt.title('Extracted corners')
        plt.axis([0,nx,ny,0])
        plt.plot(grid_pts[0,:]+1,grid_pts[1,:]+1,'r+',markersize=10,markeredgewidth=3)
        plt.plot(xorig+1,yorig+1,'*m')
        plt.text(xorig+delta*vO[0],yorig+delta*vO[1],'O',color='m',size=14)
        plt.text(dxpos[0,0]+delta*vX[0],dxpos[0,1]+delta*vX[1],'dX',color='m',size=14)
        plt.text(dypos[0,0]+delta*vY[0],dypos[0,1]+delta*vY[1],'dY',color='g',size=14)
        plt.draw()
        plt.pause(.1)
        raw_input('Press enter to continue: ')
        plt.close(imagefigure2)
        
        Xi=np.reshape((((np.mat(np.arange(0,n_sq_x+1))*dX).T)*(np.ones((1,n_sq_y+1)))).T,(Np,1)).T
        Yi=np.reshape(((np.ones((n_sq_x+1,1)))*(np.mat(np.arange(n_sq_y,-1,-1))*dY)).T,(Np,1)).T
        Zi=np.mat(np.zeros((1,Np)))
                
        
        Xgrid = np.mat(np.zeros((3,Np)))
        Xgrid[0,:]=Xi[0,:]
        Xgrid[1,:]=Yi[0,:]
        
        x=grid_pts.copy()
        #X=Xgrid.copy()
        
        
        exec('dX_'+str(kk)+'=dX') in locals(),globals()
        exec('dY_'+str(kk)+'=dY') in locals(),globals()

        exec('wintx_'+str(kk)+'=wintx') in locals(),globals()
        exec('winty_'+str(kk)+'=winty') in locals(),globals()

        exec('x_'+str(kk)+'=x') in locals(),globals()
        exec('X_'+str(kk)+'=Xgrid') in locals(),globals()

        exec('n_sq_x_'+str(kk)+'=n_sq_x') in locals(),globals()
        exec('n_sq_y_'+str(kk)+'=n_sq_y') in locals(),globals()
        
        
        active_images.append(1)
    ind_active=np.transpose(np.nonzero(active_images))
    string_save=[active_images,ind_active,wintx,winty,n_ima,nx,ny,dX_default,dY_default,dX,dY,wintx_default,winty_default]
    for kk in np.arange(1,n_ima+1):
        exec('string_save.append(X_'+str(kk)+')')
        exec('string_save.append(x_'+str(kk)+')')
        exec('string_save.append(n_sq_x_'+str(kk)+')')
        exec('string_save.append(n_sq_y_'+str(kk)+')')
        exec('string_save.append(wintx_'+str(kk)+')')
        exec('string_save.append(winty_'+str(kk)+')')
        exec('string_save.append(dX_'+str(kk)+')')
        exec('string_save.append(dY_'+str(kk)+')')
    
    with open('objs.pickle','w') as f:
        pickle.dump(string_save,f)
    
        
    
        
#########################show_window function#####################
def show_window(cell_list, fig_number, title_figure, x_size, y_size, gap_x, font_name, font_size):

    

    def but1() : print "Button was pushed" ####callback function

    
    if 'cell_list' not in locals().keys():
        print 'No description of the functions'
    
    if 'fig_number' not in locals().keys():
        fig_number = 1        
        
    if 'title_figure' not in locals().keys():
        title_figure = ''
        
    if 'x_size' not in locals().keys():
        x_size = 85
        
    if 'y_size' not in locals().keys():
        y_size = 14
    
    if 'gap_x' not in locals().keys():
        gap_x = 0
            
    if 'font_name' not in locals().keys():
        font_name='clean'
        
    if 'font_size' not in locals().keys():
        font_size = 8
    
       
    n_row=cell_list.shape[0]
    n_col=cell_list.shape[1]
    
    fig_size_x=x_size*n_col+(n_col+1)*gap_x
    fig_size_y=y_size*n_row+(n_row+1)*gap_x

    
    
    win=Tk()
    win.title("Camera Calibration Toolbox - Standard Version")
    #f= Frame(win) #small window in whole window
    pos_x=win.winfo_rootx()
    pos_y=win.winfo_rooty()
    win.geometry('%dx%d+%d+%d' % (500,40,700,300))

    
    """
    posx=np.zeros((n_row,n_col))
    posy=np.zeros((n_row,n_col))
    
    
    for i in range(1,n_row):
        for j in range(1,n_col):
            posx[n_row-i][n_col-j]=gap_x+(n_col-j)*(x_size+gap_x)
            posy[n_row-i][n_col-j]=fig_size_y - (n_row-i+1)*(gap_x+y_size)
    

    ###button instance problem
    #h_mat = np.zeros((n_row,n_col))        
    button instance problem
    for i in range(1,n_row):
        for j in range(1,n_col):
            if cell_list[n_row-i][n_col-j][0] and cell_list[n_row-i][n_col-j][1]:
                    h_mat[n_row-i][n_col-j]=Button(win, text=cell_list[n_row-i][n_col-j][0], width=x_size, command=but1)
                    h_mat[n_row-i][n_col-j].grid(row = posx[n_row-i][n_col-j], column=posy[n_row-i][n_col-j])"""
                   
    xsize=35
    ysize=2
      
    b1=Button(win,text=cell_list[0][2][0],width=xsize, height=ysize,command=eval(cell_list[0][2][1]))
    b2=Button(win,text=cell_list[0][3][0],width=xsize, height=ysize,command=eval(cell_list[0][3][1]))
    
    b1.grid(row=0, column=0)
    b2.grid(row=0, column=1)
    
    
   
    """l=Label(win, text="Camera Calibration Toolbox - Standard Version")
    l.pack()"""
    
    #win.pack()
    
    win.mainloop()

###############data_calib function##############

def data_calib():
  
    #cv2.imshow('calib_image',ima);cv2.waitKey(0)
    plt.imshow(I_1)
    plt.show()



    """print "go home"
    f = open("image_name_calib.txt",'r')
    while 1:
        line=f.readline()
        if not line: break
        ima=cv2.imread(line)
        cv2.imshow('imageshow',ima)
        cv2.waitKey(1000)
        print(line)
    f.close()"""


################################################

if __name__ == "__main__":
    cell_list=np.empty([4,4,2],dtype='S80')
    #cell_list=[[0 for col in range(4)] for row in range(4)]

    fig_number = 1

    title_figure = 'Camera Calibration Toolbox - Standard Version'

    kc = np.zeros((5,1))

    c=np.array([[0 for col in range(4)] for row in range(4)])
    cell_list[0][0] = ['Image names','data_calib']
    cell_list[0][1] = ['Read images','ima_read_calib'] #none
    cell_list[0][2] = ['Extract grid corners','click_calib']
    cell_list[0][3] = ['Calibration','go_calib_optim'] #here
    cell_list[1][0] = ['Show Extrinsic','ext_calib']
    cell_list[1][1] = ['Reproject on images','reproject_calib']
    cell_list[1][2] = ['Analyse error','analyse_error']
    cell_list[1][3] = ['Recomp. corners','recomp_corner_calib']
    cell_list[2][0] = ['Add/Suppress images','add_suppress']
    cell_list[2][1] = ['Save','saving_calib']
    cell_list[2][2] = ['Load','loading_calib']
    cell_list[2][3] = ['Exit','close']
    cell_list[3][0] = ['Comp. Extrinsic','extrinsic_computation']
    cell_list[3][1] = ['Undistort image','undistort_image']
    cell_list[3][2] = ['Export calib data','export_calib_data']
    cell_list[3][3] = ['Show calib results','show_calib_results']


    show_window(cell_list, fig_number, title_figure, 130, 18, 0, 'clean', 12)

    

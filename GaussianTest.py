import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.transforms as transform
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as annie

u0 = 4*np.pi * 10**-1 # * 10**-7
m = 10 # Sample
Bc = u0*m/4/np.pi # Constant attached to y and z components

def mapping(x0,y0,z0,n0,Ti,Te,m):
    h = 0.02 # 0.01 optimal, 0.02 fast
    a = 250000 # 250000 
    x = np.zeros(a)
    y = np.zeros(a)
    z = np.zeros(a)
    Bx = np.zeros(a)
    By = np.zeros(a)
    Bz = np.zeros(a)
    x[0] = x0
    y[0] = y0
    z[0] = z0
    Bx[0] = Bc * ((3*x0*z0)/((x0)**2+(y0)**2+(z0)**2)**(5/2))
    By[0] = Bc * ((3*y0*z0)/((x0)**2+(y0)**2+(z0)**2)**(5/2))
    Bz[0] = Bc * (3*(z0)**2-((x0)**2+(y0)**2+(z0)**2)**(1/2))/(((x0)**2+(y0)**2+(z0)**2)**(5/2))
    L = (x0**2+y0**2)**(1/2)    
    H = ((2*Ti*1.6*10**-19*(1+2*Te/Ti)/(3*m*(1.76*10**-4)**2))**(1/2))/(6.99*10**7)    
#    print(np.arcsin(z0)*180/np.pi) # Shows which starting degree is loading
    i = 1
    xL = [x0]
    yL = [y0]
    zL = [z0]
    nL = [n0]
    sL = [0]
    # Runge-Kutta 4th-order loop for a field line with a given starting point
    for i in range (1,a):
        p1 = Bc * (3*x[i-1]*z[i-1])/(((x[i-1])**2+(y[i-1])**2+(z[i-1])**2)**(5/2))
        p2 = Bc * (3*(x[i-1]+(h/2)*p1)*(z[i-1]+(h/2)*p1))/(((x[i-1]+(h/2)*p1)**2+(y[i-1]+(h/2)*p1)**2+(z[i-1]+(h/2)*p1)**2)**(5/2))
        p3 = Bc * (3*(x[i-1]+(h/2)*p2)*(z[i-1]+(h/2)*p2))/(((x[i-1]+(h/2)*p2)**2+(y[i-1]+(h/2)*p2)**2+(z[i-1]+(h/2)*p2)**2)**(5/2))
        p4 = Bc * (3*(x[i-1]+h*p3)*(z[i-1]+h*p3))/(((x[i-1]+h*p3)**2+(y[i-1]+h*p3)**2+(z[i-1]+h*p3)**2)**(5/2))
        q1 = Bc * (3*y[i-1]*z[i-1])/((((x[i-1])**2+y[i-1])**2+(z[i-1])**2)**(5/2))
        q2 = Bc * (3*(y[i-1]+(h/2)*q1)*(z[i-1]+(h/2)*q1))/(((x[i-1]+(h/2)*q1)**2+(y[i-1]+(h/2)*q1)**2+(z[i-1]+(h/2)*q1)**2)**(5/2))
        q3 = Bc * (3*(y[i-1]+(h/2)*q2)*(z[i-1]+(h/2)*q2))/(((x[i-1]+(h/2)*q2)**2+(y[i-1]+(h/2)*q2)**2+(z[i-1]+(h/2)*q2)**2)**(5/2))
        q4 = Bc * (3*(y[i-1]+h*q3)*(z[i-1])+h*q3)/(((x[i-1]+h*q3)**2+(y[i-1]+h*q3)**2+(z[i-1]+h*q3)**2)**(5/2))
        s1 = Bc * (3*((z[i-1])**2)-((x[i-1])**2+(y[i-1])**2+(z[i-1])**2))/(((x[i-1])**2+(y[i-1])**2+(z[i-1])**2)**(5/2))
        s2 = Bc * (3*((z[i-1]+(h/2)*s1)**2)-((x[i-1]+(h/2)*s1)**2+(y[i-1]+(h/2)*s1)**2+(z[i-1]+(h/2)*s1)**2))/((((x[i-1]+(h/2)*s1)**2)+(y[i-1]+(h/2)*s1)**2+(z[i-1]+(h/2)*s1)**2)**(5/2))
        s3 = Bc * (3*((z[i-1]+(h/2)*s2)**2)-((x[i-1]+(h/2)*s2)**2+(y[i-1]+(h/2)*s2)**2+(z[i-1]+(h/2)*s2)**2))/((((x[i-1]+(h/2)*s2)**2)+(y[i-1]+(h/2)*s2)**2+(z[i-1]+(h/2)*s2)**2)**(5/2))
        s4 = Bc * (3*((z[i-1]+(h*s3))**2)-((x[i-1]+h*s3)**2+(y[i-1]+h*s3)**2+(z[i-1]+h*s3)**2))/((((x[i-1]+h*s3)**2)+(y[i-1]+h*s3)**2+(z[i-1]+h*s3)**2)**(5/2))
        x[i] = x[i-1] + (h/6)*(p1 + 2*p2 + 2*p3 + p4)
        y[i] = y[i-1] + (h/6)*(q1 + 2*q2 + 2*q3 + q4)
        z[i] = z[i-1] + (h/6)*(s1 + 2*s2 + 2*s3 + s4)
        xF = x[i]
        yF = y[i]
        zF = z[i]
        sth = zF/((xF**2+yF**2+zF**2)**(1/2))
        sF = L*(3**(1/2)*np.log((3*sth**2+1)**(1/2)+3**(1/2)*sth) + 3*sth*(3*sth**2+1)**(1/2))/6
        if i % 40 == 0:
            xL.append(xF)
            yL.append(yF)
            zL.append(zF)
#           s = L*(3**(1/2)*np.log((3*sth**2+1)**(1/2)+3**(1/2)*sth) + 3*sth*(3*sth**2+1)**(1/2))/6
            sL.append(sF)
            nL.append(n0*np.exp(-(sF/(0.40*H))**2))        
        if (8.125 - ((x[i])**2 + (y[i])**2)**(1/2))**2 + (z[i])**2 > 4.52:
            break
        i = i + 1
    sP = np.array(sL)
    nP = np.array(nL)
    print(sP)
    plt.plot(sL, nL)
    
#(2*Ti*1.6*10**-19*(1+2*Te/Ti)/(3*m*(1.76*10**-4)**2))**(1/2)   
mapping(-3.425000000000003, -5.932274015923403, 0, 273.331398024509, 91.9720483640083, 5, 5.320000000000001e-26)
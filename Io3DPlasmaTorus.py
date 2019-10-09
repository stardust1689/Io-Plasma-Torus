# Started December 15th, 2018
# Last edited May 15th, 2019
# by Chris Peters

import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.transforms as transform
from mpl_toolkits.mplot3d import Axes3D

# Constants
u0 = 4*np.pi * 10**-1 # * 10**-7
m = 10 # Sample
Bc = u0*m/4/np.pi # Constant attached to x, y, and z components

# Plot definitions
fig = plt.figure(figsize=(8,6)) #edgecolor='aqua'
ax = fig.add_subplot(111, projection = '3d')
ax.set_facecolor('k')
ax.axis('scaled')
# View set for "viewed from Earth"
ax.view_init(elev=8, azim=8)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_xlim([-6,6])
ax.set_ylim([-6,6])
ax.set_zlim([-6,6])
ax.set_xlabel('x (RJ)', fontsize=14, color = 'aqua')
ax.set_ylabel('y (RJ)', fontsize=14, color = 'aqua')
ax.set_zlabel('z (RJ)', fontsize=14, color = 'aqua')
ax.set_xticklabels([])
ax.set_yticklabels(list(np.arange(-8,9,2)), color = 'aqua', rotation=0, fontsize=14)
ax.set_zticklabels(list(np.arange(-8,9,2)), color = 'aqua', rotation=0, fontsize=14)
ax.axis(aspect = ['equal'])
ax.set_title("3D Model of Io's Plasma Torus", color='aqua', fontsize=20)

# Definitions and plots for "background" radial marks
theta = np.linspace(0,2*np.pi,100)

xR = np.cos(theta)
yR = np.sin(theta)
zR = np.zeros(100)
ax.plot(2*xR,2*yR,zR, color='grey', linewidth=0.2)
ax.plot(4*xR,4*yR,zR, color='grey', linewidth=0.2)
ax.plot(8*xR,8*yR,zR, color='grey', linewidth=0.2)
ax.plot(10*xR,10*yR,zR, color='grey', linewidth=0.2)
ax.plot(12*xR,12*yR,zR, color='grey', linewidth=0.2)
ax.plot(14*xR,14*yR,zR, color='grey', linewidth=0.2)
ax.plot(16*xR,16*yR,zR, color='grey', linewidth=0.2)
ax.plot(18*xR,18*yR,zR, color='grey', linewidth=0.2)
ax.plot(20*xR,20*yR,zR, color='grey', linewidth=0.2)
ax.plot(22*xR,22*yR,zR, color='grey', linewidth=0.2)

# Function defining an individual field line in Cartesian coordinates
def Dipole(x0,y0,z0):
    h = 0.05 # 0.01
    a = 10000000 # 250000
    x = np.zeros(a)
    y = np.zeros(a)
    z = np.zeros(a)
    Bx = np.zeros(a)
    By = np.zeros(a)
    Bz = np.zeros(a)
    r = np.ones(a)
    x[0] = x0
    y[0] = y0
    z[0] = z0
    Bx[0] = Bc * ((3*x0*z0)/((x0)**2+(y0)**2+(z0)**2)**(5/2))
    By[0] = Bc * ((3*y0*z0)/((x0)**2+(y0)**2+(z0)**2)**(5/2))
    Bz[0] = Bc * (3*(z0)**2-((x0)**2+(y0)**2+(z0)**2))/(((x0)**2+(y0)**2+(z0)**2)**(5/2))
    print(np.arcsin(z0)*180/np.pi) # Shows which starting degree is loading
    i = 1
    r[0] = ((x0)**2+(y0)**2+(z0)**2)**(1/2)
    xL = [x0]
    yL = [y0]
    zL = [z0]
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
        r[i] = ((x[i])**2+ (y[i])**2 + (z[i])**2)**(1/2)
        xF = x[i]
        yF = y[i]
        zF = z[i]
        xL.append(x[i])
        yL.append(y[i])
        zL.append(z[i])
        # Loop breaks if radius is below 1, in other words the field line
        # is complete and y and z default back to zero.
        if r[i] < 1 or r[i] > 16.5 or z[i] > 13.5 or z[i] < -14.5:
            break
        i = i + 1
    # Lists turned into np.arrays
    xP = np.array(xL)
    yP = np.array(yL)
    zP = np.array(zL)
#    # y and z tilted via rotation matrix
    xPt = xP
    yPt = yP*np.cos(-7*np.pi/180) - zP*np.sin(-7*np.pi/180)
    zPt = yP*np.sin(-7*np.pi/180) + zP*np.cos(-7*np.pi/180)
    ax.plot(xPt, yPt, zPt, 'r-', alpha=1.0, linewidth=0.9)
    # y and z tilted via matplotlib.transforms method
#    base = plt.gca().transData
#    rot = transform.Affine2D().rotate_deg(-7)
#    dp = ax.plot(xP,yP,zP,'r-',transform= rot + base, alpha=1.0, linewidth=0.9)


# "Negative" function for last legs of high-angle points (b/c the number of
# steps does not allow for complete field line for angles close to 90 deg)
def DipoleNeg(x0,y0,z0):
    h = 0.02
    a = 10000000
    x = np.zeros(a)
    y = np.zeros(a)
    z = np.zeros(a)
    Bx = np.zeros(a)
    By = np.zeros(a)
    Bz = np.zeros(a)
    r = np.ones(a)
    x[0] = x0
    y[0] = y0
    z[0] = z0
    Bx[0] = - Bc * ((3*x0*z0)/((x0)**2+(y0)**2+(z0)**2)**(5/2))
    By[0] = - Bc * ((3*y0*z0)/((x0)**2+(y0)**2+(z0)**2)**(5/2))
    Bz[0] = - Bc * (3*(z0)**2-((x0)**2+(y0)**2+(z0)**2)**(1/2))/(((x0)**2+(y0)**2+(z0)**2)**(5/2))
    print(np.arcsin(z0)*180/np.pi)
    i = 1
    r[0] = ((x0)**2+(y0)**2+(z0)**2)**(1/2)
    xL = [x0]
    yL = [y0]
    zL = [z0]
    # Notice how RK4 components are negative instead of positive.
    for i in range (1,a):
        p1 = - Bc * (3*x[i-1]*z[i-1])/(((x[i-1])**2+(y[i-1])**2+(z[i-1])**2)**(5/2))
        p2 = - Bc * (3*(x[i-1]+(h/2)*p1)*(z[i-1]+(h/2)*p1))/(((x[i-1]+(h/2)*p1)**2+(y[i-1]+(h/2)*p1)**2+(z[i-1]+(h/2)*p1)**2)**(5/2))
        p3 = - Bc * (3*(x[i-1]+(h/2)*p2)*(z[i-1]+(h/2)*p2))/(((x[i-1]+(h/2)*p2)**2+(y[i-1]+(h/2)*p2)**2+(z[i-1]+(h/2)*p2)**2)**(5/2))
        p4 = - Bc * (3*(x[i-1]+h*p3)*(z[i-1]+h*p3))/(((x[i-1]+h*p3)**2+(y[i-1]+h*p3)**2+(z[i-1]+h*p3)**2)**(5/2))
        q1 = - Bc * (3*y[i-1]*z[i-1])/((((x[i-1])**2+y[i-1])**2+(z[i-1])**2)**(5/2))
        q2 = - Bc * (3*(y[i-1]+(h/2)*q1)*(z[i-1]+(h/2)*q1))/(((x[i-1]+(h/2)*q1)**2+(y[i-1]+(h/2)*q1)**2+(z[i-1]+(h/2)*q1)**2)**(5/2))
        q3 = - Bc * (3*(y[i-1]+(h/2)*q2)*(z[i-1]+(h/2)*q2))/(((x[i-1]+(h/2)*q2)**2+(y[i-1]+(h/2)*q2)**2+(z[i-1]+(h/2)*q2)**2)**(5/2))
        q4 = - Bc * (3*(y[i-1]+h*q3)*(z[i-1])+h*q3)/(((x[i-1]+h*q3)**2+(y[i-1]+h*q3)**2+(z[i-1]+h*q3)**2)**(5/2))
        s1 = - Bc * (3*((z[i-1])**2)-((x[i-1])**2+(y[i-1])**2+(z[i-1])**2))/(((x[i-1])**2+(y[i-1])**2+(z[i-1])**2)**(5/2))
        s2 = - Bc * (3*((z[i-1]+(h/2)*s1)**2)-((x[i-1]+(h/2)*s1)**2+(y[i-1]+(h/2)*s1)**2+(z[i-1]+(h/2)*s1)**2))/((((x[i-1]+(h/2)*s1)**2)+(y[i-1]+(h/2)*s1)**2+(z[i-1]+(h/2)*s1)**2)**(5/2))
        s3 = - Bc * (3*((z[i-1]+(h/2)*s2)**2)-((x[i-1]+(h/2)*s2)**2+(y[i-1]+(h/2)*s2)**2+(z[i-1]+(h/2)*s2)**2))/((((x[i-1]+(h/2)*s2)**2)+(y[i-1]+(h/2)*s2)**2+(z[i-1]+(h/2)*s2)**2)**(5/2))
        s4 = - Bc * (3*((z[i-1]+(h*s3))**2)-((x[i-1]+h*s3)**2+(y[i-1]+h*s3)**2+(z[i-1]+h*s3)**2))/((((x[i-1]+h*s3)**2)+(y[i-1]+h*s3)**2+(z[i-1]+h*s3)**2)**(5/2))
        x[i] = x[i-1] + (h/6)*(p1 + 2*p2 + 2*p3 + p4)
        y[i] = y[i-1] + (h/6)*(q1 + 2*q2 + 2*q3 + q4)
        z[i] = z[i-1] + (h/6)*(s1 + 2*s2 + 2*s3 + s4)
        r[i] = ((x[i])**2+ (y[i])**2 + (z[i])**2)**(1/2)
        xF = x[i]
        yF = y[i]
        zF = z[i]
        xL.append(x[i])
        yL.append(y[i])
        zL.append(z[i])
        if r[i] < 1 or r[i] > 16.5 or z[i] > 13.5 or z[i] < -14.5:
            break
        i = i + 1
    # Lists turned into np.arrays
    xP = np.array(xL)
    yP = np.array(yL)
    zP = np.array(zL)
    # y and z tilted via rotation matrix
    xPt = xP
    yPt = yP*np.cos(-7*np.pi/180) - zP*np.sin(-7*np.pi/180)
    zPt = yP*np.sin(-7*np.pi/180) + zP*np.cos(-7*np.pi/180)
    ax.plot(xPt, yPt, zPt, 'r-', alpha=1.0, linewidth=0.9)
#    # y and z tilted via matplotlib.transforms method
#    base = plt.gca().transData
#    rot = transform.Affine2D().rotate_deg(-7)
#    dp = ax.scatter(xP,yP,zP,'r-',transform= rot + base, alpha=1.0, linewidth=0.5)

# Jupiter
phiJ = np.linspace(0, 2 * np.pi, 25)
thetaJ = np.linspace(0, np.pi, 25)
xJ = np.outer(np.cos(phiJ), np.sin(thetaJ))
yJ = np.outer(np.sin(phiJ), np.sin(thetaJ))
zJ = np.outer(np.ones(np.size(phiJ)), np.cos(thetaJ))

# Maps and plots the density points within the torus from an initial data point 
def mapping(x0,y0,z0,n0,Ti,Te,m):
    h = 0.02 
    a = 250000 
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
    # Runge-Kutta 4th-order loop for a field line with a given starting point, similar to Dipole() function
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
        sth = zF/((xF**2+yF**2+zF**2)**(1/2)) # angle needed for arc length
        sF = L*(3**(1/2)*np.log((3*sth**2+1)**(1/2)+3**(1/2)*sth) + 3*sth*(3*sth**2+1)**(1/2))/6
        if i % 40 == 0: # if element's iteration is a multiple of 25
            # then add it to list of points to be plotted.
            xL.append(xF)
            yL.append(yF)
            zL.append(zF)
            # s = L*(3**(1/2)*np.log((3*sth**2+1)**(1/2)+3**(1/2)*sth) + 3*sth*(3*sth**2+1)**(1/2))/6
            sL.append(sF)
            nL.append(n0*np.exp(-(sF/(0.40*H))**2)) # Notice the constant in front of H that is less than one. This serves to 
            # narrow the Gaussian, making the density "seem" to drop faster, making the increased transparency more apparent.
            
        # for loop breaks if location is greater than about 2.25 RJ from inner ring of torus
        if (8.125 - ((x[i])**2 + (y[i])**2)**(1/2))**2 + (z[i])**2 > 4.52:
            break
        i = i + 1
    xP = np.array(xL)
    yP = np.array(yL)
    zP = np.array(zL)
    xPt = xP
    yPt = yP*np.cos(-7*np.pi/180) - zP*np.sin(-7*np.pi/180)
    zPt = yP*np.sin(-7*np.pi/180) + zP*np.cos(-7*np.pi/180)
    xPtNeg = xP
    yPtNeg = yP*np.cos(7*np.pi/180) - zP*np.sin(7*np.pi/180)
    zPtNeg = yP*np.sin(7*np.pi/180) + zP*np.cos(7*np.pi/180)
#    nP = np.array(nL)
    # to prevent Python from getting mad, set the nL index (to grab for alpha) to the inverse of the value that it the multiple of the plotting iteration (line 242)
    ax.scatter(xPt, yPt, zPt, alpha=1*nL[int(i/40)]/650, linewidth = 0.05, color='red', linestyle='dotted')
    ax.scatter(xPtNeg, yPtNeg, -zPtNeg, alpha=1*nL[int(i/40)]/650, linewidth = 0.05, color='red', linestyle='dotted')    
#    print((yL[-1]**2 + zL[-1]**2)**(1/2),np.arctan(zL[-1]/yL[-1])*180/np.pi)

# Opens needed .dat files and manipulates them to create lists for entry into mapping() function
file1 = open('DENSs2p0009_3D.dat')
file2 = open('TEMPs2p0009_3D.dat')
#file3 = open('TEMPelec"".dat)"
# File inputs can be changed for different ions.

mydens = []
mytemploc = []
mytemp = []
myEtemp = []

# Splits .dat files by row and turns rows into lists
for a in file1:
    mydens += [a.split()]
    
for b in file2:
    mytemploc += [b.split()]    
    
# Deletes empty lists formed from empty rows in .dat files    
del mydens[13]
del mydens[26]
del mydens[39]
del mydens[52]        
del mydens[65]        
del mydens[78]        
del mydens[91]        
del mydens[104]        
del mydens[117]        
del mydens[130]
del mydens[143]        
del mydens[156]        
del mydens[169]        
del mydens[182]        
del mydens[195]        
del mydens[208]        

del mytemploc[13]
del mytemploc[26]
del mytemploc[39]
del mytemploc[52]        
del mytemploc[65]        
del mytemploc[78]        
del mytemploc[91]        
del mytemploc[104]        
del mytemploc[117]        
del mytemploc[130]
del mytemploc[143]        
del mytemploc[156]        
del mytemploc[169]        
del mytemploc[182]        
del mytemploc[195]        
del mytemploc[208]        

# Isolates parameter representing temperature on mytemploc
for c in mytemploc:
    mytemp += [float(c[1])]
    
#for d in myEtemploc:
#    mytemp += [float(c[1])]

zubat = []

# Takes parameters from "mydens" and "mytemp" to create lists containing arguments for mapping() function
for d in mydens:
    x0 = float(d[2])*np.cos(np.pi*float(d[0])/180)
    y0 = float(d[2])*np.sin(np.pi*float(d[0])/180)
    z0 = 0
    n0 = float(d[1])
    ind = mydens.index(d)
    Ti = mytemp[ind]
    Te = 5
#    Te = myEtemp[index]
    m = 5.32*10**-26 # mass of ion 
    zubat += [[x0,y0,z0,n0,Ti,Te,m]]
    
# Execution of Dipole() and DipoleNeg() funtions mapped to Cartesian components from sherical components 
Dipole(0,np.cos(52.5*np.pi/180), np.sin(52.5*np.pi/180))
Dipole(0,np.cos(np.pi/3), np.sin(np.pi/3))
Dipole(0,np.cos(67.5*np.pi/180), np.sin(67.5*np.pi/180))
Dipole(0,np.cos(75*np.pi/180), np.sin(75*np.pi/180))
Dipole(0,np.cos(82.5*np.pi/180), np.sin(82.5*np.pi/180))
Dipole(0,0,1)
Dipole(0,np.cos(97.5*np.pi/180), np.sin(97.5*np.pi/180))
Dipole(0,np.cos(105*np.pi/180), np.sin(105*np.pi/180))
Dipole(0,np.cos(112.5*np.pi/180), np.sin(112.5*np.pi/180))
Dipole(0,np.cos(2*np.pi/3), np.sin(2*np.pi/3))
Dipole(0,np.cos(127.5*np.pi/180), np.sin(127.5*np.pi/180))

DipoleNeg(0, np.cos(255*np.pi/180), np.sin(255*np.pi/180))
DipoleNeg(0, np.cos(262.5*np.pi/180), np.sin(262.5*np.pi/180))
DipoleNeg(0,0,-1)
DipoleNeg(0, np.cos(277.5*np.pi/180), np.sin(277.5*np.pi/180))
DipoleNeg(0, np.cos(285*np.pi/180), np.sin(285*np.pi/180))

# Execution of mapping() functions for all data points mapped to spherical coordinates
for l in zubat:
    mapping(l[0],l[1],l[2],l[3],l[4],l[5],l[6])

# Orbital path for Io
rIo = 6
phiIo = np.linspace(0,8*np.pi,300)

xIo = rIo * np.cos(phiIo)
yIo = rIo * np.sin(phiIo)
zIo = np.zeros(300)
  
zRef = np.linspace(-15,15,3)  
# Miscellaneous Plots
ax.plot(np.zeros(3), np.zeros(3), zRef, color='green', label="Jupiter's Geographic Polar Axis")
ax.plot([0], [0] ,[0], color='red', label='B-Field Lines')
ax.plot(xIo, yIo ,zIo, color='pink', linewidth=0.3, label="Io's Orbital Path")
ax.plot_surface(xJ, yJ, zJ, color='sandybrown')
ax.scatter(6*np.cos(25*np.pi/180),6*np.sin(25*np.pi/180),0, color='pink')

# Legend
leg = ax.legend(loc='upper right', bbox_to_anchor=(1.34,0.937), facecolor='k', framealpha=1) #edgecolor='aqua'

for text in leg.get_texts():
    plt.setp(text, color='aqua')

plt.axis('off')
plt.show()
print('done')
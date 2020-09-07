# Started December 15th, 2018
# Last edited August 2nd, 2020
# by Chris Peters

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time
import os

t1 = time()
# Constants
u0 = 4*math.pi * 10**-1 # * 10**-7
m = 10 # Sample
Bc = u0*m/4/math.pi # Constant attached to x, y, and z components

# Plot definitions
fig = plt.figure(figsize=(6,6)) #edgecolor='aqua'
ax = fig.add_subplot(111, projection = '3d')
ax.set_facecolor('k')
# ax.axis('equal')
ax.view_init(elev=13, azim=7)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_xlim([-6.25,6.25])
ax.set_ylim([-6.25,6.25])
ax.set_zlim([-6.25,6.25])
ax.set_xlabel('x (RJ)', fontsize=14, color = 'aqua')
ax.set_ylabel('y (RJ)', fontsize=14, color = 'aqua')
ax.set_zlabel('z (RJ)', fontsize=14, color = 'aqua')
ax.set_xticklabels([])
ax.set_yticklabels(list(np.arange(-8,9,2)), color = 'aqua', rotation=0, fontsize=14)
ax.set_zticklabels(list(np.arange(-8,9,2)), color = 'aqua', rotation=0, fontsize=14)
ax.set_title("3D Model of Io's Plasma Torus (S 3+)", color='aqua', fontsize=15)

# Definitions and plots for "background" radial marks
theta = np.linspace(0,2*math.pi,100)

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
ax.plot([-15,15],[0,0],[0,0], color='grey', linewidth=0.2)
ax.plot([0,0],[-15,15],[0,0], color='grey', linewidth=0.2)

# Maps and plots the density points within the torus from an initial data point 
def mapping(mapping_data):
    x0 = mapping_data[0]
    y0 = mapping_data[1]
    z0 = mapping_data[2]
    n0 = mapping_data[3]
    Ti = mapping_data[4]
    Te = mapping_data[5]
    m = mapping_data[6]
    h = 0.02 
    a = 250000 
    x = [0] * a
    y = [0] * a
    z = [0] * a
    x[0] = x0
    y[0] = y0
    z[0] = z0
    # Bx = Bc * ((3*x0*z0)/((x0)**2+(y0)**2+(z0)**2)**(5/2))
    # By = Bc * ((3*y0*z0)/((x0)**2+(y0)**2+(z0)**2)**(5/2))
    # Bz = Bc * (3*(z0)**2-((x0)**2+(y0)**2+(z0)**2)**(1/2))/(((x0)**2+(y0)**2+(z0)**2)**(5/2))
    L = (x0**2+y0**2)**(1/2)    
    H = ((2*Ti*1.6*10**-19*(1+2*Te/Ti)/(3*m*(1.76*10**-4)**2))**(1/2))/(6.99*10**7)       
    i = 1
    x_list = [x0]
    y_list = [y0]
    z_list = [z0]
    n_list = [n0/2500]
    s_list = [0]  
    
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
        sth = z[i]/((x[i]**2+y[i]**2+z[i]**2)**(1/2)) # angle needed for arc length
        s = L*(3**(1/2)*math.log((3*sth**2+1)**(1/2)+3**(1/2)*sth) + 3*sth*(3*sth**2+1)**(1/2))/6
        if i % 225 == 0: 
            # if element's iteration is a multiple of 225
            # then add it to list of points to be plotted. (Reduces plotting time)
            x_list.append(x[i])
            y_list.append(y[i])
            z_list.append(z[i])
            s_list.append(s)
            n_list.append((n0*math.exp(-(s/(0.65*H))**2))/2500) 
            # Notice the constant in front of H that is less than one. This serves to 
            # narrow the Gaussian, making the density "seem" to drop faster, making the increased transparency more apparent.
            # The greater the coefficient, the less of the "drop."
            
        # "for" loop breaks if location is greater than about 2.25 RJ from the center of the torus.
        if (8.125 - ((x[i])**2 + (y[i])**2)**(1/2))**2 + (z[i])**2 > 4.52:
            break
        i = i + 1
        
    return([x_list, y_list, z_list, n_list])
    
# Plots a series of equi-radial data points and their mappings
def map_to_plot(input_data):
    print(f"Mapping for radius {round(input_data[0][7], 2)} rJ (Jupiter radii)")
    radius_map = []
    
    # Populates radius_map with mapping() results from input_data argument
    for mapping_data in input_data:
        radius_map.append(mapping(mapping_data))
    
    # lengths list is built so the minimum length can be accessed, fixing the error of nonexistent elements from unequal length lists
    lengths = []
    for n in range(len(radius_map)):
        lengths.append(len(radius_map[n][0])) 
    
    # Creates series of points to plot together
    for m in range(min(lengths)):
        x_rad = []
        y_rad = []
        z_rad = []
        n_rad = []
        
        # Lists populated then tilted for magnetic equatorial mapping
        for n in range(len(radius_map)):
            x_rad.append(radius_map[n][0][m])
            y_rad.append(radius_map[n][1][m])
            z_rad.append(radius_map[n][2][m])
            n_rad.append(radius_map[n][3][m])
            x_rad_tilt = np.array(x_rad)
            y_rad_tilt = np.array(y_rad)*np.cos(-7*np.pi/180) - np.array(z_rad)*np.sin(-7*np.pi/180)
            z_rad_tilt = np.array(y_rad)*np.sin(-7*np.pi/180) + np.array(z_rad)*np.cos(-7*np.pi/180)
            x_rad_tilt_neg = np.array(x_rad)
            y_rad_tilt_neg = np.array(y_rad)*np.cos(7*np.pi/180) - np.array(z_rad)*np.sin(7*np.pi/180)
            z_rad_tilt_neg = -(np.array(y_rad)*np.sin(7*np.pi/180) + np.array(z_rad)*np.cos(7*np.pi/180))

        # Tilted list (now np.arrays) plotted with lines connecting points by adjacent angle
        for index in range(len(x_rad_tilt)-1):
            # alpha (transparency of lines) to be the mean density of the two points, adjusted to be between 0 and 1
            alpha = (n_rad[index] + n_rad[index+1]) / 2
            ax.plot(x_rad_tilt[index:index+2], y_rad_tilt[index:index+2], z_rad_tilt[index:index+2], color='yellow', alpha=alpha, linewidth=2.5)
            if n != 0:
                ax.plot(x_rad_tilt_neg[index:index+2], y_rad_tilt_neg[index:index+2], z_rad_tilt_neg[index:index+2], color='yellow', alpha=alpha, linewidth=2.5)
            
        alpha = (n_rad[-1] + n_rad[0]) / 2
        ax.plot([x_rad_tilt[-1], x_rad_tilt[0]],[y_rad_tilt[-1], y_rad_tilt[0]], [z_rad_tilt[-1], z_rad_tilt[0]], color = 'yellow', alpha=alpha, linewidth=2.5)
        ax.plot([x_rad_tilt_neg[-1], x_rad_tilt_neg[0]],[y_rad_tilt_neg[-1], y_rad_tilt_neg[0]], [z_rad_tilt_neg[-1], z_rad_tilt_neg[0]], color = 'yellow', alpha=alpha, linewidth=2.5)

b_field_x = []
b_field_y = []
b_field_z = []

# Takes data from .dat files created from MagFieldLines.py and plots field lines, done to make repetitive use of program faster
# Without .dat files, program will raise error. For first time use, run MagFieldLines.py before this program.
# After using MagFieldLines.py, you never need to use it again (unless this or .dat files are moved / modified).
for data_file in os.listdir('./B_Field_Data'):
    data = open('./B_Field_Data/' + data_file)
    if "_x" in data_file:
        b_x_line = []
        for text_line in data:
            b_x_line.append(float(text_line))
        b_field_x.append(b_x_line)
    elif "_y" in data_file:
        b_y_line = []
        for text_line in data:
            b_y_line.append(float(text_line))
        b_field_y.append(b_y_line)
    elif "_z" in data_file:
        b_z_line = []
        for text_line in data:
            b_z_line.append(float(text_line))
        b_field_z.append(b_z_line)
    
for n in range(len(b_field_x)):
    plt.plot(b_field_x[n], b_field_y[n], b_field_z[n], 'r-', linewidth=0.9)
    

# Opens needed .dat files and manipulates them to create lists for entry into map_to_plot() function
# file1 = open('./data/plots/data/{x}/DENS/DENS{x}{y}_3D.dat')
# file2 = open('./data/plots/data/{x}/TEMP/TEMP{x}{y}_3D.dat')
# x = any from following: op, o2p, sp, s2p, s3p, elec
# y = any number from 0000 to 0200 (MUST include zeros)
file1 = open('./data/plots/data/s3p/DENS/DENSs3p0145_3D.dat')
file2 = open('./data/plots/data/s3p/TEMP/TEMPs3p0145_3D.dat')
#file3 = open('TEMPelec.dat')
# File inputs can be changed for different ions.

my_dens = []
my_temp_loc = []
my_temp = []
my_e_temp = []

# Splits .dat files by row and turns rows into lists
for a in file1:
    my_dens += [a.split()]
    
for b in file2:
    my_temp_loc += [b.split()]    
    
# Deletes empty lists formed from empty rows in .dat files so that Python will not find something wrong with the universe
for n in range(1,209):
    if n % 13 == 0:
        del my_dens[n]
        del my_temp_loc[n]
    
# Isolates parameter representing temperature on my_temp_loc
for c in my_temp_loc:
    my_temp += [float(c[1])]

zubat = []

# Takes parameters from "my_dens" and "my_temp" to create data lists for one point at a time
for d in my_dens:
    x0 = float(d[2])*math.cos(math.pi*float(d[0])/180)
    y0 = float(d[2])*math.sin(math.pi*float(d[0])/180)
    z0 = 0
    n0 = float(d[1])
    index = my_dens.index(d)
    Ti = my_temp[index]
    Te = 5
#    Te = my_e_temp[index]
    m = 5.32 * 10**-26 # mass of ion 
    r0 = math.sqrt(x0**2 + y0**2)
    zubat += [[x0, y0, z0, n0, Ti, Te, m, r0]]

# Combines "zubat" lists to a series of data for one RADIUS at a time
golbat = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
p = 0

for n in range(0, 208, 13):
    for m in range(0,12):
        golbat[p].append(zubat[m+n])
    p += 1

# # Execution of map_to_plot() functions for all data points mapped to Cartesian coordinates
# for n in range(len(golbat)):
#     crobat = golbat[n]
#     map_to_plot(crobat)
  
zRef = np.linspace(-15,15,3)

# Jupiter
phiJ = np.linspace(0, 2 * np.pi, 25)
thetaJ = np.linspace(0, np.pi, 25)
xJ = np.outer(np.cos(phiJ), np.sin(thetaJ))
yJ = np.outer(np.sin(phiJ), np.sin(thetaJ))
zJ = np.outer(np.ones(np.size(phiJ)), np.cos(thetaJ))
 
# Miscellaneous Plots
ax.plot([0] * 3, [0] * 3, zRef, color='green', label="Jupiter's Geographic Polar Axis")
ax.plot([0], [0] ,[0], color='red', label='Magnetic Field Lines')
ax.plot_surface(xJ, yJ, zJ, color='sandybrown')

# Legend
leg = ax.legend(loc='upper right', bbox_to_anchor=(1,0), facecolor='k', framealpha=1) #edgecolor='aqua'

for text in leg.get_texts():
    plt.setp(text, color='aqua')

plt.axis('off')

plt.show()
t2 = time()
print(f'done in {t2-t1} seconds')

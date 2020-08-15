import numpy as np
import math

# Constants
u0 = 4*math.pi * 10**-1 # * 10**-7
m = 10 # Sample
Bc = u0*m/4/math.pi # Constant attached to x, y, and z components

# Function defining an individual field line in Cartesian coordinates
def dipole(x0,y0,z0,degrees):
    h = 0.05
    a = 10000000
    x = [0] * a
    y = [0] * a
    z = [0] * a
    r = [1] * a
    x[0] = x0
    y[0] = y0
    z[0] = z0
    # Bx = Bc * ((3*x0*z0)/((x0)**2+(y0)**2+(z0)**2)**(5/2))
    # By = Bc * ((3*y0*z0)/((x0)**2+(y0)**2+(z0)**2)**(5/2))
    # Bz = Bc * (3*(z0)**2-((x0)**2+(y0)**2+(z0)**2))/(((x0)**2+(y0)**2+(z0)**2)**(5/2))
    if y0 < 0:
        print(f"{180 - round(math.degrees(math.asin(z0)))} degrees")
    else:
        print(f"{round(math.degrees(math.asin(z0)))} degrees") # Shows which starting degree is loading
    i = 1
    r[0] = ((x0)**2+(y0)**2+(z0)**2)**(1/2)
    x_list = [x0]
    y_list = [y0]
    z_list = [z0]
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
        x_list.append(x[i])
        y_list.append(y[i])
        z_list.append(z[i])
        # Loop breaks if radius is below 1, in other words the field line
        # is complete and y and z default back to zero.
        if r[i] < 1 or r[i] > 16.5 or z[i] > 13.5 or z[i] < -14.5:
            break
        i = i + 1

    # coordinates tilted via rotation matrix
    x_tilt = np.array(x_list)
    y_tilt = np.array(y_list)*np.cos(np.radians(-7)) - np.array(z_list)*np.sin(np.radians(-7))
    z_tilt = np.array(y_list)*np.sin(np.radians(-7)) + np.array(z_list)*np.cos(np.radians(-7))
    # Arrays saved into .dat files for use in Io3DPlasmaTorus.py
    np.savetxt(f"./B_Field_Data/{degrees}_x.dat", x_tilt)
    np.savetxt(f"./B_Field_Data/{degrees}_y.dat", y_tilt)
    np.savetxt(f"./B_Field_Data/{degrees}_z.dat", z_tilt)

# "Negative" function for last legs of high-angle points (b/c the number of
# steps does not allow for complete field line for angles close to +90 deg)
def dipole_neg(x0,y0,z0,degrees):
    h = 0.02
    a = 10000000
    x = [0] * a
    y = [0] * a
    z = [0] * a
    r = [1] * a
    x[0] = x0
    y[0] = y0
    z[0] = z0
    # Bx = - Bc * ((3*x0*z0)/((x0)**2+(y0)**2+(z0)**2)**(5/2))
    # By = - Bc * ((3*y0*z0)/((x0)**2+(y0)**2+(z0)**2)**(5/2))
    # Bz = - Bc * (3*(z0)**2-((x0)**2+(y0)**2+(z0)**2)**(1/2))/(((x0)**2+(y0)**2+(z0)**2)**(5/2))
    if y0 < 0:
        print(f"{-(180 + round(math.degrees(math.asin(z0))))} degrees")
    else:
        print(f"{round(math.degrees(math.asin(z0)))} degrees") # Shows which starting degree is loading
    i = 1
    r[0] = ((x0)**2+(y0)**2+(z0)**2)**(1/2)
    x_list = [x0]
    y_list = [y0]
    z_list = [z0]
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
        x_list.append(x[i])
        y_list.append(y[i])
        z_list.append(z[i])
        if r[i] < 1 or r[i] > 16.5 or z[i] > 13.5 or z[i] < -14.5:
            break
        i = i + 1
        
    x_tilt = np.array(x_list)
    y_tilt = np.array(y_list)*np.cos(-7*np.pi/180) - np.array(z_list)*np.sin(-7*np.pi/180)
    z_tilt = np.array(y_list)*np.sin(-7*np.pi/180) + np.array(z_list)*np.cos(-7*np.pi/180)
    np.savetxt(f"./B_Field_Data/{degrees}_x.dat", x_tilt)
    np.savetxt(f"./B_Field_Data/{degrees}_y.dat", y_tilt)
    np.savetxt(f"./B_Field_Data/{degrees}_z.dat", z_tilt)

# Execution of dipole() and dipole_neg() funtions mapped to Cartesian components from sherical components 
dipole(0,math.cos(math.radians(52.5)), math.sin(math.radians(52.5)), "52point5")
dipole(0,math.cos(math.radians(60)), math.sin(math.radians(60)), "60")
dipole(0,math.cos(math.radians(67.5)), math.sin(math.radians(67.5)), "67point5")
dipole(0,math.cos(math.radians(75)), math.sin(math.radians(75)), "75")
dipole(0,math.cos(math.radians(82.5)), math.sin(math.radians(82.5)), "82point5")
dipole(0,0,1, "90")
dipole(0,math.cos(math.radians(97.5)), math.sin(math.radians(97.5)), "97point5")
dipole(0,math.cos(math.radians(105)), math.sin(math.radians(105)), "105")
dipole(0,math.cos(math.radians(112.5)), math.sin(math.radians(112.5)), "112point5")
dipole(0,math.cos(math.radians(120)), math.sin(math.radians(120)), "120")
dipole(0,math.cos(math.radians(127.5)), math.sin(math.radians(127.5)), "127point5")

dipole_neg(0, math.cos(math.radians(255)), math.sin(math.radians(255)), "255")
dipole_neg(0, math.cos(math.radians(262.5)), math.sin(math.radians(262.5)), "262point5")
dipole_neg(0,0,-1, "270")
dipole_neg(0, math.cos(math.radians(277.5)), math.sin(math.radians(277.5)), "277point5")
dipole_neg(0, math.cos(math.radians(285)), math.sin(math.radians(285)), "285")
print("Magnetic dipole field lines done")
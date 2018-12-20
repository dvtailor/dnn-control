#####################################################################################
#
# Problem:  Optimal Control Problem (OCP)
# Dynamics: Point mass in 2D with one reaction wheel controlling the pitch rate and
#           one main thruster (in essence: a quad rotor)
# Transcription: Hermite-Simpson
#
# Author: Dario Izzo (Nov 2012)
#
#####################################################################################

#Parameters---------------------------------------------------
#Generic
    param n default 30;             #Number of nodes
    param gearth:=9.81;             #[m/s^2] Earth gravity constant

#Quadrotor params
    param mass default 0.38905;        #[kg] Mass
    param maxthrust default 9.1;       #[N] Max thrust
    param minthrust default 0;         #[N] Min thrust
    param maxthetadot default 35;      #[rad/s] Max pitch rate

#Initial conditions
    param x0 default 1.0;        #[m] Initial x
    param z0 default 1.0;        #[m] Initial z
    param vx0 default 0.0;       #[m/s] Initial velocity in x
    param vz0 default 0.0;       #[m/s] Initial Velocity in z
    param theta0 default 0.0;    #[rad] Initial pitch

#Final conditions
    param xn default 0.0;        #[m] Final x
    param zn default 0.0;        #[m] Final z
    param vxn default 0.0;       #[m/s] Final velocity in x
    param vzn default 0.0;       #[m/s] Final velocity z
    param thetan default 0.0;    #[rad] Final pitch

#Other
    param tn:=3.0;                    #[s] Guess for the final time
#-------------------------------------------------------------

#Sets---------------------------------------------------------
    set I := 1..n;
    set J := 1..n-1;
#-------------------------------------------------------------

#Variables---------------------------------------------------
    var x{i in I};
    var vx{i in I};
    var z{i in I};
    var vz{i in I};
    var theta{i in I};
    var u1{i in I}, >=0;
    var u2{i in I};
    var u1m{i in J}, >=0;
    var u2m{i in J};
#-------------------------------------------------------------

#Time variables-----------------------------------------------
    var tf, >=0;
    var dt = tf/(n-1);
    var timegrid{i in I} = dt*(i-1);
#-------------------------------------------------------------

#Objective----------------------------------------------------

        # For power, minimize Simpson's approximation to the integral:
        #
        #        \int{ f(t)dt }
        #     ~= \sum_{  dt/6 * f(t) + 4*f(t+dt/2)  + f(t+dt)  }
        #               for t=(dt,2*dt,3*dt...)

        # The cost function is:
        #  f(t) = alpha * u1(t)^2 + u2(t)^2
        # The weight alpha indicates the relative importance of u1 vs u2

        #cost has the values at t = i*dt
        #cost_m has the values at t = i*dt + dt/2
    var cost{i in I} = (u1[i]*u1[i] + u2[i]*u2[i]);
    var cost_m{i in J} = (u1m[i]*u1m[i] + u2m[i]*u2m[i]);
    var power = dt/6 * sum{i in J} (cost[i]+4*cost_m[i]+cost[i+1]);

    minimize myobjective: power;

#-------------------------------------------------------------

#Dynamic at the grid points-----------------------------------
    var f1{i in I} = vx[i];
    var f2{i in I} = u1[i] * sin(theta[i]) / mass - vx[i]/2;
    var f3{i in I} = vz[i];
    var f4{i in I} = u1[i] * cos(theta[i]) / mass - gearth - vz[i]/2;
    var f5{i in I} = u2[i];
#-----------------------------------------------------------------------

#State definition at mid-points via Hermite interpolation---------------
    var xm{i in J}      =   (x[i] + x[i+1])/2 + tf/(n-1)/8 * (f1[i] - f1[i+1]);
    var vxm{i in J}     =   (vx[i] + vx[i+1])/2 + tf/(n-1)/8 * (f2[i] - f2[i+1]);
    var zm{i in J}      =   (z[i] + z[i+1])/2 + tf/(n-1)/8 * (f3[i] - f3[i+1]);
    var vzm{i in J}     =   (vz[i] + vz[i+1])/2 + tf/(n-1)/8 * (f4[i] - f4[i+1]);
    var thetam{i in J}  =   (theta[i] + theta[i+1])/2 + tf/(n-1)/8 * (f5[i] - f5[i+1]);
#-----------------------------------------------------------------------

#Dynamic at the mid-points----------------------------------------------
    var f1m{i in J} = vxm[i];
    var f2m{i in J} = u1m[i]*sin(thetam[i])/mass - vxm[i]/2;
    var f3m{i in J} = vzm[i];
    var f4m{i in J} = u1m[i]*cos(thetam[i])/mass - gearth - vzm[i]/2;
    var f5m{i in J} = u2m[i];
#-----------------------------------------------------------------------

#Simpson Formula---------------------------------------------------------
subject to
    dynamicx{i in J}:  x[i]  = x[i+1] - tf/(n-1)/6*(f1[i] + f1[i+1] + 4*f1m[i]);
    dynamicvx{i in J}:  vx[i]  = vx[i+1]  - tf/(n-1)/6*(f2[i] + f2[i+1] + 4*f2m[i]);
    dynamicz{i in J}:  z[i]  = z[i+1]  - tf/(n-1)/6*(f3[i] + f3[i+1] + 4*f3m[i]);
    dynamicvz{i in J}: vz[i] = vz[i+1] - tf/(n-1)/6*(f4[i] + f4[i+1] + 4*f4m[i]);
    dynamictheta{i in J}: theta[i] = theta[i+1] - tf/(n-1)/6*(f5[i] + f5[i+1] + 4*f5m[i]);
#--------------------------------------------------------------------------

#Constraints------------------------------------------
    #Boundary Conditions
    subject to InitialPositionx: x[1] = x0;
    subject to InitialPositionz: z[1] = z0;
    subject to InitialVelocityx: vx[1] = vx0;
    subject to InitialVelocityz: vz[1] = vz0;
    subject to InitialPitch: theta[1] = theta0;

    #Final
    subject to FinalPositionx: x[n] = xn;
    subject to FinalPositionz: z[n] = zn;
    subject to FinalVelocityz: vz[n] = vzn;
    subject to FinalVelocityx: vx[n] = vxn;
    subject to FinalPitch: theta[n] = thetan;


    #Control Constraint
    #Control Magnitude Constraint
    subject to controlmagnitude1{i in I}: u1[i] - maxthrust <=0;
    subject to controlmagnitude2{i in I}: u1[i] - minthrust >=0;
    subject to controlmagnitude3{i in I}: u2[i] - maxthetadot <=0;
    subject to controlmagnitude4{i in I}: u2[i] + maxthetadot >=0;

    subject to controlmagnitude1m{i in J}: u1m[i] - maxthrust <=0;
    subject to controlmagnitude2m{i in J}: u1m[i] - minthrust >=0;
    subject to controlmagnitude3m{i in J}: u2m[i] - maxthetadot <=0;
    subject to controlmagnitude4m{i in J}: u2m[i] + maxthetadot >=0;
#-------------------------------------------------------------

#Guess-------------------------------------------------------
    let tf := tn;
    #let {i in I} vx[i] := vx0;
    #let {i in I} x[i] := x0 + vx[i]*i*dt;
    #let {i in I} u1[i] :=maxthrust * tn/(n-2);
    #let {i in I} u2[i] :=maxthrust * tn/(n-2);
#-------------------------------------------------------------

#Solver Options-----------------------------------------------
    option solver snopt;
    option substout 0;
    option show_stats 1;
    /* option reset_initial_guesses 1; */
    options ipopt_options "outlev=2";
    options snopt_options "outlev=2 Major_iterations=2000 Superbasics=1500";
#-------------------------------------------------------------

# #Solve!!!-----------------------------------------------------
#     solve;
# #-------------------------------------------------------------

# #Print the Solution with midpoints---------------------------
#     for {i in J}
#     {
#     printf "%17.16e %17.16e %17.16e %17.16e %17.16e %17.16e %17.16e %17.16e\n", timegrid[i],  x[i], vx[i], z[i], vz[i], theta[i], u1[i], u2[i] > out/sol_power.out;
#     printf "%17.16e %17.16e %17.16e %17.16e %17.16e %17.16e %17.16e %17.16e\n", timegrid[i] + dt/2, xm[i], vxm[i], zm[i], vzm[i], thetam[i], u1m[i], u2m[i] > out/sol_power.out;
#     }
#     printf "%17.16e %17.16e %17.16e %17.16e %17.16e %17.16e %17.16e %17.16e\n", timegrid[n],  x[n], vx[n], z[n], vz[n], theta[n], u1[n], u2[n] > out/sol_power.out;
#     close out/sol_power.out;
# #-------------------------------------------------------------

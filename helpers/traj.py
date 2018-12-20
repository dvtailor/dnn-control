from .shared_imports import *

import io, pickle
from amplpy import AMPL
from contextlib import redirect_stdout

state_idx = {'time':0, 'x':1, 'vx':2, 'z':3, 'vz':4, 'theta':5, 'u1':6, 'u2':7}
state_vars = ['x','z','vx','vz','theta']


def get_trajectory(params, ampl_mdl_path, hide_solver_output=True):
    ampl = AMPL() # ampl installation directory should be in system search path
    # .mod file
    ampl.read(ampl_mdl_path)

    # set parameter values
    for lbl, val in params.items():
        _ampl_set_param(ampl, lbl, val)

    _ampl_solve(ampl, hide_solver_output)
    optimal_sol_found = _ampl_optimal_sol_found(ampl)

    traj = None
    objective = None
    if optimal_sol_found:
        traj = _extract_trajectory_from_solver(ampl)
        objective = ampl.getObjective('myobjective').value()

    ampl.close()
    return optimal_sol_found, traj, objective


# solves for regularised objective starting from non-regularised solution
def get_trajectory_regularised(params, ampl_mdl_path, hide_solver_output=True):
    ampl = AMPL()
    ampl.read(ampl_mdl_path)

    for lbl, val in params.items():
        _ampl_set_param(ampl, lbl, val)

    _ampl_set_param(ampl, 'reg', 0) # no regularisation
    _ampl_solve(ampl, hide_solver_output)

    optimal_sol_found = _ampl_optimal_sol_found(ampl)

    traj = None
    objective = None
    optimal_sol_found_reg = False
    if optimal_sol_found:
        _ampl_set_param(ampl, 'reg', 1) # regularisation
        _ampl_solve(ampl, hide_solver_output)
        optimal_sol_found_reg = _ampl_optimal_sol_found(ampl)

        if optimal_sol_found_reg:
            traj = _extract_trajectory_from_solver(ampl)
            objective = ampl.getObjective('myobjective').value()

    ampl.close()
    return optimal_sol_found_reg, traj, objective


def _ampl_set_param(ampl, lbl, val):
    ampl.getParameter(lbl).set(val)


def _ampl_solve(ampl, hide_solver_output):
    if hide_solver_output:
        with redirect_stdout(None):
            ampl.solve()
    else:
        ampl.solve()


def _ampl_optimal_sol_found(ampl):
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        ampl.display('solve_result')

    result_status = stdout.getvalue().rstrip()
    optimal_sol_found = (result_status == 'solve_result = solved')
    return optimal_sol_found


def _extract_trajectory_from_solver(ampl):
    # extract required variables from ampl
    nodes = int(ampl.getParameter('n').get())
    timegrid = [val for (_,val) in ampl.getVariable('timegrid').getValues().toList()]
    dt = ampl.getVariable('dt').getValues().toList()[0]

    x_arr = [val for (_,val) in ampl.getVariable('x').getValues().toList()]
    xm_arr = [val for (_,val) in ampl.getVariable('xm').getValues().toList()]
    vx_arr = [val for (_,val) in ampl.getVariable('vx').getValues().toList()]
    vxm_arr = [val for (_,val) in ampl.getVariable('vxm').getValues().toList()]
    z_arr = [val for (_,val) in ampl.getVariable('z').getValues().toList()]
    zm_arr = [val for (_,val) in ampl.getVariable('zm').getValues().toList()]
    vz_arr = [val for (_,val) in ampl.getVariable('vz').getValues().toList()]
    vzm_arr = [val for (_,val) in ampl.getVariable('vzm').getValues().toList()]
    theta_arr = [val for (_,val) in ampl.getVariable('theta').getValues().toList()]
    thetam_arr = [val for (_,val) in ampl.getVariable('thetam').getValues().toList()]
    u1_arr = [val for (_,val) in ampl.getVariable('u1').getValues().toList()]
    u1m_arr = [val for (_,val) in ampl.getVariable('u1m').getValues().toList()]
    u2_arr = [val for (_,val) in ampl.getVariable('u2').getValues().toList()]
    u2m_arr = [val for (_,val) in ampl.getVariable('u2m').getValues().toList()]

    # number of points along trajectory = 2*nodes-1
    traj_arr = np.zeros(shape=(2*nodes-1, 8))
    for i in np.arange(nodes-1):
        traj_arr[2*i,:] = np.asarray([timegrid[i],
                                      x_arr[i],
                                      vx_arr[i],
                                      z_arr[i],
                                      vz_arr[i],
                                      theta_arr[i],
                                      u1_arr[i],
                                      u2_arr[i]])
        traj_arr[2*i+1,:] = np.asarray([timegrid[i] + dt/2.0,
                                        xm_arr[i],
                                        vxm_arr[i],
                                        zm_arr[i],
                                        vzm_arr[i],
                                        thetam_arr[i],
                                        u1m_arr[i],
                                        u2m_arr[i]])
    traj_arr[-1,:] = np.asarray([timegrid[-1],
                                 x_arr[-1],
                                 vx_arr[-1],
                                 z_arr[-1],
                                 vz_arr[-1],
                                 theta_arr[-1],
                                 u1_arr[-1],
                                 u2_arr[-1]])
    return traj_arr


'''
    n_repeats is the number of OCPs that will be attempted
    final number of optimal trajectories will be <= n_repeats
    n_jobs is number of CPUs to use
'''
def generate_trajectories(n_repeats, init_conds_range, ampl_mdl_path,
                            quad_params={}, n_jobs=1, get_trajectory=get_trajectory):
    # rnd number generators use half-open interval
    # instead use next float after upper limit to give U[low, high]
    get_upper_limit = lambda high : np.nextafter(high, high+1)

    # sample initial conditions
    x0 = np.random.uniform(low=init_conds_range['x'][0],
                           high=get_upper_limit(init_conds_range['x'][1]),
                           size=(n_repeats,1))
    z0 = np.random.uniform(low=init_conds_range['z'][0],
                           high=get_upper_limit(init_conds_range['z'][1]),
                          size=(n_repeats,1))
    vx0 = np.random.uniform(low=init_conds_range['vx'][0],
                            high=get_upper_limit(init_conds_range['vx'][1]),
                           size=(n_repeats,1))
    vz0 = np.random.uniform(low=init_conds_range['vz'][0],
                            high=get_upper_limit(init_conds_range['vz'][1]),
                           size=(n_repeats,1))
    theta0 = np.random.uniform(low=init_conds_range['theta'][0],
                               high=get_upper_limit(init_conds_range['theta'][1]),
                              size=(n_repeats,1))

    # construct list of params
    init_conds_arr = np.concatenate((x0,vx0,z0,vz0,theta0), axis=1)
    init_conds_lst = [init_conds_arr[i,:] for i in range(n_repeats)]
    ampl_params_lst = [{**{'x0':x0, 'z0':z0, 'vx0':vx0, 'vz0':vz0, 'theta0':theta0},
                        **quad_params}
                        for (x0,vx0,z0,vz0,theta0) in init_conds_lst]

    # solve trajectories in parallel
    sol_lst = Parallel(n_jobs)(delayed(get_trajectory)(params, ampl_mdl_path) for params in ampl_params_lst)

    # indices where solution found
    opt_sol_found_lst = [sol[0] for sol in sol_lst]
    indices_lst = [idx for idx, elem in enumerate(opt_sol_found_lst) if elem==True]

    # filter out elems where no solution found
    init_conds_arr_filtered = init_conds_arr[indices_lst]
    trajs = [sol[1] for sol in sol_lst]
    traj_arr = np.array([trajs[idx] for idx in indices_lst])
    objs = [sol[2] for sol in sol_lst]
    objs_arr = np.array([objs[idx] for idx in indices_lst])

    return traj_arr, init_conds_arr_filtered, objs_arr


# indices_lst indicates successful solutions to OCP
# trajs, objs are lists where None entries indicate unsuccessful solutions
def generate_trajectories_given_initial_conds(init_conds_arr, ampl_mdl_path, quad_params={},
                                                n_jobs=1, get_trajectory=get_trajectory):
    init_conds_lst = [init_conds_arr[i,:] for i in range(len(init_conds_arr))]

    ampl_params_lst = [{**{'x0':x0, 'z0':z0, 'vx0':vx0, 'vz0':vz0, 'theta0':theta0},
                    **quad_params}
                    for (x0,vx0,z0,vz0,theta0) in init_conds_lst]

    sol_lst = Parallel(n_jobs)(delayed(get_trajectory)(params, ampl_mdl_path) for params in ampl_params_lst)

    opt_sol_found_lst = [sol[0] for sol in sol_lst]
    indices_lst = [idx for idx, elem in enumerate(opt_sol_found_lst) if elem==True]

    trajs = [sol[1] for sol in sol_lst]
    objs = [sol[2] for sol in sol_lst]

    return trajs, objs, indices_lst


def filter_trajs(trajs_tuple, obj_lim=500):
    trajs, init_conds_arr, objs = trajs_tuple
    indices = np.where(objs < obj_lim)[0]
    return trajs[indices], init_conds_arr[indices], objs[indices]

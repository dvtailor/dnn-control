from helpers.shared_imports import *
from helpers.integrate import state_indices
from helpers.traj import state_idx, get_trajectory, get_trajectory_regularised

import pygmo, random


def compute_quadratic_cost(traj, idx):
    return traj[idx,state_idx['u1']]**2 + traj[idx,state_idx['u2']]**2


def compute_quadratic_obj(traj):
    dt = traj[1,0] - traj[0,0]
    return dt/2 * sum([(compute_quadratic_cost(traj, i) + compute_quadratic_cost(traj, i+1)) \
                                 for i in range(len(traj)-1)])


# compute optimal trajectory for new final state
def get_trajectory_with_new_final_state(initial_state, final_state, get_trajectory, ampl_path):
    [x0, vx0, z0, vz0, theta0] = initial_state.astype(np.float64)
    [xf, vxf, zf, vzf, thetaf] = final_state.astype(np.float64)
    params = {'x0' : x0, 'z0' : z0, 'vx0' : vx0, 'vz0' : vz0, 'theta0' : theta0,
              'xn' : xf, 'zn' : zf, 'vxn' : vxf, 'vzn' : vzf, 'thetan' : thetaf}
    return get_trajectory(params, ampl_path)


def get_objective_relative_error(traj_nn, ampl_path):
    optimal_sol_found, traj_ocp, obj_ocp = \
        get_trajectory_with_new_final_state(traj_nn[0,state_indices], traj_nn[-1,state_indices], get_trajectory, ampl_path)

    obj_rel_err = None
    if optimal_sol_found:
        if obj_ocp < 100: # objective limit in dataset generation (bebop, power)
            obj_nn = compute_quadratic_obj(traj_nn)
            obj_rel_err = (obj_nn - obj_ocp) * 100 / obj_ocp
        else:
            optimal_sol_found = False

    if not optimal_sol_found:
        traj_nn = None
        traj_ocp = None

    return optimal_sol_found, obj_rel_err, traj_nn, traj_ocp


def get_tf_relative_error(traj_nn, ampl_path):
    optimal_sol_found, traj_ocp, _ = \
        get_trajectory_with_new_final_state(traj_nn[0,state_indices], traj_nn[-1,state_indices], \
                                            get_trajectory_regularised, ampl_path)

    tf_rel_error = None
    if optimal_sol_found:
        tf_nn = traj_nn[-1,0]
        tf_ocp = traj_ocp[-1,0]
        tf_rel_error = (tf_nn - tf_ocp) * 100 / tf_ocp

    return optimal_sol_found, tf_rel_error, traj_nn, traj_ocp


# bebop only
def compute_objective_relative_errors(traj_sims_lst, obj_type='power', n_jobs=45):
    if obj_type == 'power':
        rel_err_func = get_objective_relative_error
        ampl_mdl_path = os.path.expanduser('~/Documents/neurocontroller-hotm/ampl/bebop_power.mod')
    elif obj_type == 'time':
        rel_err_func = get_tf_relative_error
        ampl_mdl_path = os.path.expanduser('~/Documents/neurocontroller-hotm/ampl/bebop_time.mod')
    else:
        raise ValueError('unrecognised argument for obj_type')

    sol_lst = Parallel(n_jobs)(delayed(rel_err_func)(traj, ampl_mdl_path) \
                                 for traj in traj_sims_lst)

    index_tups = [(idx, x[1], x[2], x[3]) for idx, x in enumerate(sol_lst) if x[0]]
    indices_succ, rel_obj_vals, trajs_nn, trajs_ocp = zip(*index_tups)

    return np.array(rel_obj_vals), list(indices_succ), list(trajs_nn), list(trajs_ocp)


# factor : percentage increase on total number of timesteps to simulate
def truncate_traj_sims(traj_sims, final_times, dt=0.01, factor=0.1):
    indices = np.ceil(final_times * (1.0/dt) * (1.0+factor)).astype('int')
    return [traj_sims[i, :indices[i]+1] for i in range(len(traj_sims))]


def get_simulation_cutoff(traj_sims, tol_pos, tol_vel, tol_theta, equilibrium=None):
    state_tols = [tol_pos,tol_vel,tol_pos,tol_vel,tol_theta]

    if equilibrium is None:
        equilibrium = np.array([0,0,0,0,0])

    satisfy_tol_bool_arr = \
        np.logical_and.reduce(np.absolute(traj_sims[:,:,state_indices] - equilibrium) < state_tols, axis=2)

    satisfy_tol_indices_arr = np.asarray(np.where(satisfy_tol_bool_arr)).T
    satisfy_tol_indices_partition_arr = \
        np.split(satisfy_tol_indices_arr,
                 np.unique(satisfy_tol_indices_arr[:,0], return_index=True)[1][1:], axis=0)

    # there exists a trajectory for which no points along it satisfy tolerance
    if len(satisfy_tol_indices_partition_arr) < len(traj_sims):
        raise ValueError('trajectory found that does not meet tolerance')

    indices_tol_first_reached = np.array([np.min(x[:,1]) for x in satisfy_tol_indices_partition_arr])
    traj_sims_lst = [traj_sims[i, :indices_tol_first_reached[i]+1] for i in range(len(traj_sims))]

    return traj_sims_lst


'''
find equilibrium point using EA
state search space restricted to {x,z}, others fixed at 0
'''
def get_equilibrium(nn_controller, random_seed=0, mass=0.38905):
    class controller_equilibrium:
        def __init__(self, nn_controller):
            self.nn_controller = nn_controller

        def fitness(self, state):
            g0 = 9.81
            return [np.linalg.norm(self.nn_controller.compute_control(state) - np.array([g0*mass, 0]))]

        def get_bounds(self):
            return ([-1,0,-1,0,0],[1,0,1,0,0])

    # optimisation parameters
    # increase if necessary to ensure convergence
    n_generations = 700
    n_individuals = 100

    prob = pygmo.problem(controller_equilibrium(nn_controller))
    algo = pygmo.algorithm(pygmo.de(gen=n_generations, tol=0, ftol=0, seed=random_seed))

    pop = pygmo.population(prob, size=n_individuals, seed=random_seed)
    pop.push_back([0,0,0,0,0])
    algo.set_verbosity(100)
    pop = algo.evolve(pop)

    return pop.champion_x

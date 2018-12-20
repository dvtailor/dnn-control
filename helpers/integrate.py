from .shared_imports import *

import itertools, pickle
from functools import partial
from pyaudi import gdual_double, sin, cos, tanh, exp, log
from scipy.integrate import odeint

time_idx = 0
state_indices = [1,2,3,4,5]
control_indices = [6,7]
mass_bebop = 0.38905


# compatible with gduals
def simulate_system(nn_controller, d_state, initial_state, \
                    mass=1, dt=0.01, max_time=5.0):
    n_steps = int(max_time / dt) + 1
    traj_sim = np.zeros((n_steps, 8), dtype=gdual_double)

    traj_sim[0,state_indices] = initial_state
    traj_sim[0,control_indices] = nn_controller.compute_control(initial_state)

    dx = lambda state : d_state(state, nn_controller.compute_control(state), mass)

    # necessary to avoid operating directly on numpy array
    # since array is not standard type (gdual) code runs sub-optimally compared with float
    # even if actual data type is not gdual
    state = initial_state
    for i in range(1, n_steps):
        traj_sim[i,time_idx] = i*dt

        state = _runge_kutta_odeint(dx, state, dt)
        traj_sim[i,state_indices] = state
        traj_sim[i,control_indices] = nn_controller.compute_control(state)

    return traj_sim


# default error tolerance 1.49012e-8
# from https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
def simulate_system_odeint(nn_controller, d_state, initial_state, \
                           mass=1, dt=0.01, max_time=10, err_tol=None):
    time_steps = np.arange(0, np.nextafter(max_time, max_time+1), dt)
    dx = lambda state, t: d_state(state, nn_controller.compute_control(state), mass)

    states = odeint(dx, initial_state, time_steps, rtol=err_tol, atol=err_tol)
    controls = np.array([nn_controller.compute_control(states[i]) \
                         for i in range(len(states))])

    traj = np.zeros((len(time_steps), 8))
    traj[:,0] = time_steps
    traj[:,1:6] = states
    traj[:,6:] = controls

    return traj


'''
simulate with delayed controls
we use scipy's odeint instead of runge-kutta implementation below
because we do not need to be compatible with gduals
faster than delayed solver below since controls are kept constant between
time steps (computed from states at time steps; no interpolation)
if use small time step (1e-3) then trajectory equivalent to dde solver
'''
def simulate_system_delay(nn_controller, d_state, initial_state, mass=1, \
                            dt=0.001, max_time=5.0, delay_step=0):
    n_steps = int(max_time / dt) + 1
    traj_sim = np.zeros((n_steps, 8))

    # t < tau then (1) control is u(x0) ; (2) control is 0
    get_control = lambda i : nn_controller.compute_control(traj_sim[0,state_indices]) if (i < 0) \
                                else nn_controller.compute_control(traj_sim[i,state_indices])
    # get_control = lambda i : np.array([0,0]) if (i < 0) \
    #                             else nn_controller.compute_control(traj_sim[i,state_indices])

    traj_sim[0,state_indices] = initial_state
    traj_sim[0,control_indices] = get_control(0-delay_step)

    # control constant for time step interval
    dx = lambda state, t, controls : d_state(state, controls, mass)

    for i in range(1, n_steps):
        traj_sim[i,time_idx] = i*dt

        state = traj_sim[i-1,state_indices]
        traj_sim[i,state_indices] = odeint(dx, state, [0, dt], args=(traj_sim[i-1,control_indices],))[1]

        traj_sim[i,control_indices] = get_control(i-delay_step)

    return traj_sim


'''
simulation using delayed differential equation (DDE) solver
adaptive RK scheme with cubic Hermite interpolation for delayed variable calculation
'''
def simulate_system_dde(nn_controller, d_state, initial_state, mass=1,
                            dt=0.01, max_time=5.0, delay_step=0, err_tol=1e-8):
    # restricting pydde import to function scope fixes bug when delay_step=0
    from PyDDE import pydde

    def odegrad(s, c, t):
        time_delay = c[0]
        if (t > time_delay):
            state_delay = np.zeros_like(s)
            for i in range(len(s)):
                state_delay[i] = pydde.pastvalue(i, t-time_delay, 0)
            controls = nn_controller.compute_control(state_delay)
        else:
            controls = nn_controller.compute_control(initial_state) # np.zeros((2,))
        return d_state(s, controls, mass)

    ode_solver = pydde.dde()
    time_delay = delay_step * dt

    n_steps = int(max_time / dt) + 1
    traj_sim = np.zeros((n_steps, 8))

    states = ode_solver.dde(y=initial_state,
                            times=np.arange(0,
                                    np.nextafter(max_time,max_time+1),
                                    dt),
                            func=odegrad,
                            parms=np.array([time_delay]),
                            dt=dt,
                            nlag=1,
                            tol=err_tol)

    controls = np.array([nn_controller.compute_control(states[i,1:]) \
                          for i in range(len(states))])

    traj_sim = np.zeros((len(states), 8))
    traj_sim[:,:6] = states

    if delay_step > 0:
        traj_sim[:delay_step,6:] = nn_controller.compute_control(initial_state) # np.zeros((2,))
        traj_sim[delay_step:,6:] = controls[:-delay_step]
    else:
        traj_sim[:,6:] = controls

    return traj_sim


def _runge_kutta_odeint(dx, state, dt):
    f1 = dx(state)
    x1 = state + (dt/2.0)*f1

    f2 = dx(x1)
    x2 = state + (dt/2.0)*f2

    f3 = dx(x2)
    x3 = state + dt*f3

    f4 = dx(x3)
    return state + (dt/6.0)*(f1 + 2*f2 + 2*f3 + f4)


def d_state(state, controls, mass=1):
    [x, vx, z, vz, theta] = state
    [u1, u2] = controls

    g = 9.81

    d_x = vx
    d_vx = u1 * sin(theta) / mass
    d_z = vz
    d_vz = u1 * cos(theta) / mass - g
    d_theta = u2

    return np.array([d_x, d_vx, d_z, d_vz, d_theta])


def d_state_bebop(state, controls, mass=0.38905):
    [x, vx, z, vz, theta] = state
    [u1, u2] = controls

    g = 9.81

    d_x = vx
    d_vx = u1 * sin(theta) / mass - vx/2
    d_z = vz
    d_vz = u1 * cos(theta) / mass - g - vz/2
    d_theta = u2

    return np.array([d_x, d_vx, d_z, d_vz, d_theta])


def simulate_linear_system(jacobian_mtx, d_state, initial_state, mass=1, \
                            dt=0.01, max_time=10, err_tol=None):
    time_steps = np.arange(0, np.nextafter(max_time, max_time+1), dt)
    dx = lambda state, t : jacobian_mtx.dot(state)

    states = odeint(dx, initial_state, time_steps)

    traj = np.zeros((len(time_steps), 6))
    traj[:,0] = time_steps
    traj[:,state_indices] = states

    return traj


def simulate_linear_delay_system(jacobian_matrices, d_state, initial_state, mass=1, \
                                     dt=0.001, max_time=10, delay_step=0):
    n_steps = int(max_time / dt) + 1
    traj_sim = np.zeros((n_steps, 6)) # t, state vector
    traj_sim[0, state_indices] = initial_state

    A,B = jacobian_matrices
    dx = lambda state, t, state_delayed : A.dot(state) + B.dot(state_delayed)

    get_delayed_state = lambda i : initial_state if (i < 0) \
                            else traj_sim[i, state_indices]

    for i in range(1, n_steps):
        traj_sim[i,time_idx] = i*dt

        state = traj_sim[i-1,state_indices]
        state_delayed = get_delayed_state(i-1-delay_step)

        traj_sim[i,state_indices] = odeint(dx, state, [0, dt], args=(state_delayed,))[1]

    return traj_sim


'''
    some numpy operations use BLAS which executes in parallel
    faster to restrict to single-threaded when simulating multiple trajs in parallel
    run os.environ['OMP_NUM_THREADS'] = '1' before importing numpy
'''
def simulate_system_multiple(nn_controller, d_state, initial_states, \
                             mass=1, dt=0.01, max_time=10, n_jobs=20):
    initial_states_lst = [initial_states[i,:] for i in range(len(initial_states))]
    simulate = partial(simulate_system_odeint, nn_controller, d_state, mass=mass, dt=dt, max_time=max_time)
    sims_lst = Parallel(n_jobs)(delayed(simulate)(state) for state in initial_states_lst)
    return np.array(sims_lst, dtype=np.float32)


def get_unbiased_controller(nn_controller, bias):
    class UnbiasedController:
        def __init__(self, nn_controller, bias):
            self.nn_controller = nn_controller
            self.bias = bias

        def compute_control(self, state):
            return self.nn_controller.compute_control(state+self.bias)

    return UnbiasedController(nn_controller, bias)

# nn_controller should be unbiased
def get_stabilised_controller(nn_controller, bias):
    # Gains for the PD controller (hand tuned)
    kpz = 2.0  # proportional term on z
    kdz = 0.7  # derivative term on z
    kpt = 10   # proportional term on theta
    kpp = 0.2  # proportional term on auxiliary
    kdp = 0.2  # derivative term on auxiliary

    # squared neighbourhood radius for using the PD
    scaling_coeff = 0.5

    # PD controller
    def nested_control(state):
        [x, vx, z, vz, theta] = state
        g0 = 9.81
        m = 0.38905

        phi = -kpp * x - kdp * vx
        uz = g0 * m - kpz * z - kdz * vz
        ut = kpt * (phi - theta)

        return np.array([uz,ut])

    # overall controller (patching NN)
    def patched_control(state, control):
        dist_sq = sum([x**2 for x in state])
        c = exp(-scaling_coeff/dist_sq)
        return control * c + (1-c) * nested_control(state)

    class StabilisedController:
        def __init__(self, nn_controller, bias):
            self.nn_controller = nn_controller
            self.bias = bias

        def compute_control(self, state):
            control = self.nn_controller.compute_control(state+self.bias)
            return patched_control(state+self.bias, control)

    return StabilisedController(nn_controller, bias)


# computes jacobian matrix of dynamics at x=0
# assumes controller is unbiased i.e. f(0)=0
# bebop drag dynamics only
def get_linearised_dynamics(controller):
    state_lbls = ['x','vx','z','vz','th']
    x_e = [gdual_double(0., lbl, 1) for lbl in state_lbls]

    delta_x = d_state_bebop(x_e, controller.compute_control(x_e))
    for y in delta_x:
        y.extend_symbol_set(['d'+lbl for lbl in state_lbls])

    n_states = len(state_indices)
    jacobian_mtx = np.zeros((n_states,n_states))
    for i in range(n_states):
        for j in range(n_states):
            jacobian_mtx[i,j] = delta_x[i].get_derivative({'d'+state_lbls[j] : 1})

    return jacobian_mtx


# computes jacobian matrices of delay dynamics f(x,xtau) at x=0
# assumes controller is unbiased i.e. f(0)=0
# bebop drag dynamics only
def get_linearised_delay_dynamics(controller):
    state_lbls = ['x','vx','z','vz','th']
    x_e = [gdual_double(0., lbl, 1) for lbl in state_lbls]
    xtau_e = [gdual_double(0., lbl+'t', 1) for lbl in state_lbls]

    delta_x = d_state_bebop(x_e, controller.compute_control(xtau_e))
    for y in delta_x:
        y.extend_symbol_set(list(itertools.chain(*[['d'+lbl, 'd'+lbl+'t'] for lbl in state_lbls])))

    n_states = len(state_indices)

    jacobian_mtx = np.zeros((n_states,n_states))
    for i in range(n_states):
        for j in range(n_states):
            jacobian_mtx[i,j] = delta_x[i].get_derivative({'d'+state_lbls[j] : 1})

    jacobian_mtx_tau = np.zeros((n_states,n_states))
    for i in range(n_states):
        for j in range(n_states):
            jacobian_mtx_tau[i,j] = delta_x[i].get_derivative({'d'+state_lbls[j]+'t' : 1})

    return jacobian_mtx, jacobian_mtx_tau


'''
    Works with numpy arrays of floats and gduals
'''
class Controller:
    def __init__(self, model=None, scalers=None, path_to_pickle=None):
        if path_to_pickle is None:
            input_scaler = scalers[0]
            self.input_scaler_params = (input_scaler.mean_, input_scaler.scale_)

            output_normaliser = scalers[1]
            output_scaler = scalers[2]
            self.output_scaler_params = (output_scaler.min_,
                                         output_scaler.scale_,
                                         output_normaliser.scale_,
                                         output_normaliser.mean_)

            self.config = model.get_config()
            self.weights = model.get_weights()
        else:
            self.input_scaler_params, self.output_scaler_params, self.config, self.weights \
                = pickle.load(open(path_to_pickle, 'rb'))

    def preprocess_input(self, state):
        return (state - self.input_scaler_params[0]) / self.input_scaler_params[1]

    def postprocess_output(self, pred):
        out = (pred - self.output_scaler_params[0]) / self.output_scaler_params[1]
        return (out * self.output_scaler_params[2]) + self.output_scaler_params[3]

    def nn_predict(self, model_input):
        vector = model_input
        dense_layer_count = 0
        for layer_config in self.config:
            if layer_config['class_name'] == 'Dense':
                wgts, biases = self.weights[dense_layer_count*2 : (dense_layer_count+1)*2]
                vector = wgts.T.dot(vector) + biases
                dense_layer_count += 1
            elif layer_config['class_name'] == 'Activation':
                if layer_config['config']['activation'] == 'relu':
                    vector[convert_gdual_to_float(vector) < 0] = 0
                elif layer_config['config']['activation'] == 'tanh':
                    vector = np.vectorize(lambda x : tanh(x))(vector)
                elif layer_config['config']['activation'] == 'softplus':
                    # avoiding overflow with exp; | log(1+exp(x)) - x | < 1e-10   for x>=30
                    floatize = lambda x : x.constant_cf if type(x) == gdual_double else x
                    softplus = lambda x : x if floatize(x) > 30.0 else log(exp(x)+1)
                    # softplus = lambda x : log(exp(x)+1)
                    vector = np.vectorize(softplus)(vector)
        return vector

    def compute_control(self, state):
        model_input = self.preprocess_input(state)
        model_pred = self.nn_predict(model_input)
        control_out = self.postprocess_output(model_pred)
        return control_out

    def convert_gdual_to_float(gdual_array):
        floatize = lambda x : x.constant_cf if type(x) == gdual_double else x
        convert_to_float = np.vectorize(floatize, otypes=[np.float64])
        return convert_to_float(gdual_array)

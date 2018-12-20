from .shared_imports import *
from .traj import state_idx

import matplotlib.animation as animation


def trajectory_plot(traj, state_only=False):
    fig = plt.figure(figsize=(10, 7))

    plt.subplot(331)
    plt.plot(traj[:,state_idx['x']], traj[:,state_idx['z']])
    plt.xlabel('$x$ (m)')
    plt.ylabel('$z$ (m)')

    plt.subplot(332)
    plt.plot(traj[:,state_idx['time']], traj[:,state_idx['x']])
    plt.xlabel('time (s)')
    plt.ylabel('$x$ (m)')

    plt.subplot(333)
    plt.plot(traj[:,state_idx['time']], traj[:,state_idx['z']])
    plt.xlabel('time (s)')
    plt.ylabel('$z$ (m)')

    plt.subplot(334)
    plt.plot(traj[:,state_idx['time']], traj[:,state_idx['vx']])
    plt.xlabel('time (s)')
    plt.ylabel('$v_x$ (m/s)')

    plt.subplot(335)
    plt.plot(traj[:,state_idx['time']], traj[:,state_idx['vz']])
    plt.xlabel('time (s)')
    plt.ylabel('$v_z$ (m/s)')

    plt.subplot(336)
    plt.plot(traj[:,state_idx['time']], traj[:,state_idx['theta']])
    plt.xlabel('time (s)')
    plt.ylabel('$\Theta$ (rad)')

    if not state_only:
        plt.subplot(337)
        plt.plot(traj[:,state_idx['time']], traj[:,state_idx['u1']])
        plt.xlabel('time (s)')
        plt.ylabel('$u_1$ (N)')

        plt.subplot(338)
        plt.plot(traj[:,state_idx['time']], traj[:,state_idx['u2']])
        plt.xlabel('time (s)')
        plt.ylabel('$u_2$ (rad/s)')

    plt.tight_layout()
    # plt.close(fig)

    return fig


def trajectory_plot_compare(traj1, traj2=None, label1='1', label2='2', state_only=False):
    fig = plt.figure(figsize=(10,7))

    plt.subplot(331)
    if traj2 is not None:
        plt.plot(traj2[:, state_idx['x']], traj2[:, state_idx['z']], label=label2)
    plt.plot(traj1[:, state_idx['x']], traj1[:, state_idx['z']], label=label1)
    plt.xlabel('$x$ (m)')
    plt.ylabel('$z$ (m)')
    plt.legend()

    plt.subplot(332)
    if traj2 is not None:
        plt.plot(traj2[:, state_idx['time']], traj2[:, state_idx['x']])
    plt.plot(traj1[:, state_idx['time']], traj1[:, state_idx['x']])
    plt.xlabel('time (s)')
    plt.ylabel('$x$ (m)')

    plt.subplot(333)
    if traj2 is not None:
        plt.plot(traj2[:, state_idx['time']], traj2[:, state_idx['z']])
    plt.plot(traj1[:, state_idx['time']], traj1[:, state_idx['z']])
    plt.xlabel('time (s)')
    plt.ylabel('$z$ (m)')

    plt.subplot(334)
    if traj2 is not None:
        plt.plot(traj2[:, state_idx['time']], traj2[:, state_idx['vx']])
    plt.plot(traj1[:, state_idx['time']], traj1[:, state_idx['vx']])
    plt.xlabel('time (s)')
    plt.ylabel('$v_x$ (m/s)')

    plt.subplot(335)
    if traj2 is not None:
        plt.plot(traj2[:, state_idx['time']], traj2[:, state_idx['vz']])
    plt.plot(traj1[:, state_idx['time']], traj1[:, state_idx['vz']])
    plt.xlabel('time (s)')
    plt.ylabel('$v_z$ (m/s)')

    plt.subplot(336)
    if traj2 is not None:
        plt.plot(traj2[:, state_idx['time']], traj2[:, state_idx['theta']])
    plt.plot(traj1[:, state_idx['time']], traj1[:, state_idx['theta']])
    plt.xlabel('time (s)')
    plt.ylabel('$\Theta$ (rad)')

    if not state_only:
        plt.subplot(337)
        if traj2 is not None:
            plt.plot(traj2[:, state_idx['time']], traj2[:, state_idx['u1']], label=label2)
        plt.plot(traj1[:, state_idx['time']], traj1[:, state_idx['u1']], label=label1)
        plt.xlabel('time (s)')
        plt.ylabel('$u_1$ (N)')
        plt.legend()

        plt.subplot(338)
        if traj2 is not None:
            plt.plot(traj2[:, state_idx['time']], traj2[:, state_idx['u2']])
        plt.plot(traj1[:, state_idx['time']], traj1[:, state_idx['u2']])
        plt.xlabel('time (s)')
        plt.ylabel('$u_2$ (rad/s)')

    plt.tight_layout()
    # plt.close(fig)

    return fig


# default arguments for bebop
# frame delay is time in milliseconds between frames (or rows in traj_arr)
def trajectory_animate(traj_arr, pos_lim=15, max_time=3.5, thrust_max=9.1, theta_max=35, frame_delay=50):
    fig = plt.figure(figsize=(9,5))

    xs = traj_arr[:,1]
    zs = traj_arr[:,3]
    vxs = traj_arr[:,2]
    vzs = traj_arr[:,4]
    thetas = traj_arr[:,5]
    times = traj_arr[:,0]
    u1s = traj_arr[:,6]
    u2s = traj_arr[:,7]

    # coordinates for quad visualisation
    vis_quad_len = 1
    xs_1 = xs + vis_quad_len * np.cos(thetas)
    xs_2 = xs - vis_quad_len * np.cos(thetas)
    zs_1 = zs - vis_quad_len * np.sin(thetas)
    zs_2 = zs + vis_quad_len * np.sin(thetas)

    ax = fig.add_subplot(121, autoscale_on=False, xlim=(-pos_lim-1.0, pos_lim+1.0), ylim=(-pos_lim-1.0, pos_lim+1.0))
    ax.grid()

    quad, = ax.plot([], [], 'o-', lw=2)
    traj, = ax.plot([], [], '-', lw=2)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    ax = fig.add_subplot(222, autoscale_on=False, xlim=(0, max_time), ylim=(0, thrust_max+0.5))
    thrust, = ax.plot([], [], '-')
    ax.set_xlabel('t')
    ax.set_ylabel('thrust (N)')

    ax = fig.add_subplot(224, autoscale_on=False, xlim=(0, max_time), ylim=(-theta_max-2.0, theta_max+2.0))
    torque, = ax.plot([], [], '-')
    ax.set_xlabel('t')
    ax.set_ylabel('angular velocity (rad/s)')

    fig.tight_layout()

    def init():
        quad.set_data([], [])
        traj.set_data([],[])
        thrust.set_data([], [])
        torque.set_data([], [])
        time_text.set_text('')

    def animate(i):
        quad.set_data([xs_1[i], xs_2[i]], [zs_1[i], zs_2[i]])
        traj.set_data(xs[:i], zs[:i])
        thrust.set_data(times[:i], u1s[:i])
        torque.set_data(times[:i], u2s[:i])
        # time_text.set_text('time = {.2f}s'.format(times[i])) # not working

    anim = animation.FuncAnimation(fig, animate, np.arange(1, len(times)),
                                  interval=frame_delay, blit=True, repeat=False, init_func=init)

    return anim

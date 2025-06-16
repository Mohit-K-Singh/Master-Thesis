import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from dolfinx import fem

def plot_solution(x_coords, wavefunctions, sorted_indices):
    
    t = np.linspace(0, 4*np.pi, len(wavefunctions))
    fig, ax = plt.subplots()
    line_real, = ax.plot([], [], label="Re(ψ)", color="blue")
    line_imag, = ax.plot([], [], label="Im(ψ)", color="orange")
    line_prob, = ax.plot([], [], label="|ψ|²", color="green")

    ax.set_xlim(x_coords.min(), x_coords.max())
    ax.set_ylim(-1.2, 1.2)
    ax.set_title("Time-dependent Schrödinger Equation")
    ax.set_xlabel("x")
    ax.set_ylabel("Amplitude")
    ax.legend()

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    def init():
        line_real.set_data([], [])
        line_imag.set_data([], [])
        line_prob.set_data([], [])
        time_text.set_text('')
        return line_real, line_imag, line_prob, time_text

    def animate(i):
        psi_vals = wavefunctions[i][sorted_indices]
        line_real.set_data(x_coords, np.real(psi_vals))
        line_imag.set_data(x_coords, np.imag(psi_vals))
        line_prob.set_data(x_coords, np.abs(psi_vals)**2)
        time_text.set_text(f'Time t = {t[i]:.2f}')
        return line_real, line_imag, line_prob, time_text

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(wavefunctions),
        interval=50, blit=True
    )

    plt.show()
    ani.save("true_1d.gif", writer="pillow")

def plot_solution_together(x_coords, wavefunctions, sorted_indices,x2_coords, wavefunctions2, sorted2_indices):
     
    t = np.linspace(0, 4*np.pi, len(wavefunctions))
    fig, ax = plt.subplots()
    line_real, = ax.plot([], [], label="Re(ψ)", color="blue")
    line_imag, = ax.plot([], [], label="Im(ψ)", color="orange")
    line_prob, = ax.plot([], [], label="|ψ|²", color="green")

    line_real_ex, = ax.plot([], [], label="Re(ψ_exact)", linestyle="--", color="blue")
    line_imag_ex, = ax.plot([], [], label="Im(ψ_exact)", linestyle="--", color="orange")
    line_prob_ex, = ax.plot([], [], label="|ψ_exact|²", linestyle="--", color="green")

    ax.set_xlim(x_coords.min(), x_coords.max())
    ax.set_ylim(-1.2, 1.2)
    ax.set_title("Time-dependent Schrödinger Equation")
    ax.set_xlabel("x")
    ax.set_ylabel("Amplitude")
    ax.legend()

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    def init():
        for line in (line_real, line_imag, line_prob,
                     line_real_ex, line_imag_ex, line_prob_ex):
            line.set_data([], [])
        time_text.set_text('')
        return (line_real, line_imag, line_prob,
                line_real_ex, line_imag_ex, line_prob_ex, time_text)

    def animate(i):
        psi_vals = wavefunctions[i][sorted_indices]
        line_real.set_data(x_coords, np.real(psi_vals))
        line_imag.set_data(x_coords, np.imag(psi_vals))
        line_prob.set_data(x_coords, np.abs(psi_vals)**2)
        time_text.set_text(f'Time t = {t[i]:.2f}')


        psi_ex_vals = wavefunctions2[i][sorted2_indices]
        line_real_ex.set_data(x2_coords, np.real(psi_ex_vals))
        line_imag_ex.set_data(x2_coords, np.imag(psi_ex_vals))
        line_prob_ex.set_data(x2_coords, np.abs(psi_ex_vals)**2)
        return line_real, line_imag, line_prob, line_prob_ex, line_imag_ex, line_prob_ex, time_text

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(wavefunctions),
        interval=50, blit=True
    )

    plt.show()
    ani.save("solution_together.gif", writer="pillow")


def plot_solution_with_exact(x_coords, wavefunctions, sorted_indices,x2_coords, wavefunctions2, sorted2_indices, L2, H1):
    t = np.linspace(0, 4*np.pi, len(wavefunctions))

    # Create two subplots: numerical (left), exact (right)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Numerical solution
    ax_num = axs[0]
    line_real_num, = ax_num.plot([], [], label="Re(ψ_h)", color="blue")
    line_imag_num, = ax_num.plot([], [], label="Im(ψ_h)", color="orange")
    line_prob_num, = ax_num.plot([], [], label="|ψ_h|²", color="green")
    ax_num.set_title(fr"Numerical Solution | L2 :{L2:.4f} | H1: {H1:.4f}")
    ax_num.set_xlim(x_coords.min(), x_coords.max())
    ax_num.set_ylim(-1.2, 1.2)
    ax_num.set_xlabel("x")
    ax_num.set_ylabel("y")
    ax_num.legend()
    time_text = ax_num.text(0.02, 0.95, '', transform=ax_num.transAxes)

    # Exact solution
    ax_exact = axs[1]
    line_real_ex, = ax_exact.plot([], [], label="Re(ψ_exact)",  color="blue")
    line_imag_ex, = ax_exact.plot([], [], label="Im(ψ_exact)", color="orange")
    line_prob_ex, = ax_exact.plot([], [], label="|ψ_exact|²", color="green")
    ax_exact.set_title("Analytical Solution")
    ax_exact.set_xlim(x2_coords.min(), x2_coords.max())
    ax_exact.set_xlabel("x")
    ax_exact.legend()


    def init():
        for line in (line_real_num, line_imag_num, line_prob_num,
                     line_real_ex, line_imag_ex, line_prob_ex):
            line.set_data([], [])
        time_text.set_text('')
        return (line_real_num, line_imag_num, line_prob_num,
                line_real_ex, line_imag_ex, line_prob_ex, time_text)

    def animate(i):

        # Numerical
        psi_vals = wavefunctions[i][sorted_indices]
        line_real_num.set_data(x_coords, np.real(psi_vals))
        line_imag_num.set_data(x_coords, np.imag(psi_vals))
        line_prob_num.set_data(x_coords, np.abs(psi_vals)**2)

        # Exact
        psi_ex_vals = wavefunctions2[i][sorted2_indices]
        line_real_ex.set_data(x2_coords, np.real(psi_ex_vals))
        line_imag_ex.set_data(x2_coords, np.imag(psi_ex_vals))
        line_prob_ex.set_data(x2_coords, np.abs(psi_ex_vals)**2)

        time_text.set_text(f'Time t = {t[i]:.2f}')
        return (line_real_num, line_imag_num, line_prob_num,
                line_real_ex, line_imag_ex, line_prob_ex, time_text)

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(wavefunctions),
        interval=50, blit=True
    )

    plt.tight_layout()
    plt.show()
    ani.save("comparison_1d_column.gif", writer="pillow")
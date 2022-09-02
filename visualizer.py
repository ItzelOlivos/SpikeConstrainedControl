import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np
from scipy.stats import norm
from matplotlib.animation import FuncAnimation


def create_demo(T, x, mu, sig, y, r, M, ri, N, ro, l):
    # ===================  Aux variables ===================
    T_max = 1
    pad = 10
    neurons_in = np.arange(0, M)
    neurons_rec = np.arange(0, N)
    steps = np.arange(0, T)
    sqrt_sig = np.sqrt(sig)
    sqrt_r = np.sqrt(r)
    dot_size = 8
    max_ri = np.max(ri)
    max_ro = np.max(ro)

    # ===================  Color maps ======================
    custom_cmap_i = plt.get_cmap("hot")
    rescale_i = lambda neurons_in: (neurons_in - np.min(neurons_in)) / (np.max(neurons_in) - np.min(neurons_in))

    custom_cmap_o = plt.get_cmap("gnuplot2")
    rescale_o = lambda neurons_rec: (neurons_rec - np.min(neurons_rec)) / (np.max(neurons_rec) - np.min(neurons_rec))

    # ================= Setting subplots ===================
    fig, axs = plt.subplots(2, 5, gridspec_kw={'width_ratios': [3, .5, 3, .5, 1.5]}, figsize=[12, 7])
    gs = axs[1, 0].get_gridspec()
    for ax in axs[1, :]:
        ax.remove()
    world_ax = fig.add_subplot(gs[1:, :])
    plt.subplots_adjust(wspace=0.4,
                        hspace=0.3)

    def clear():
        world_ax.clear()
        for ax in axs[0, :]:
            ax.clear()

        arrow = image.imread('arrow.png')
        axs[0, 1].imshow(arrow, extent=(-1, 1.5, 1.25, 1.75), zorder=-1, aspect='auto')
        axs[0, 1].set_xlim([0., 3])
        axs[0, 1].set_ylim([0., 3])
        axs[0, 1].axis('off')
        axs[0, 3].imshow(arrow, extent=(-1, 1.5, 1.25, 1.75), zorder=-1, aspect='auto')
        axs[0, 3].set_xlim([0., 3])
        axs[0, 3].set_ylim([0., 3])
        axs[0, 3].axis('off')
        # fig.tight_layout()

        # ===================  Labels ==========================
        # Random walk
        world_ax.set_xlim([0, T + pad])
        world_ax.set_xlabel('time')
        world_ax.set_ylabel('state')
        world_ax.set_ylim([-3*np.std(x), 3*np.std(x)])
        domain = np.linspace(world_ax.get_ylim()[0], world_ax.get_ylim()[1], 500)

        # Decoding axis
        axs[0, 4].axis('off')
        axs[0, 4].set_title('Decoding')

        # Sensory layer
        axs[0, 0].set_xlabel('spikes per second')
        axs[0, 0].set_ylabel('Neurons Sensory Layer')
        axs[0, 0].set_xlim([0, max_ri])
        axs[0, 0].set_title('Encoding')

        # Recurrent layer
        axs[0, 2].set_xlabel('spikes per second')
        axs[0, 2].set_ylabel('Neurons Recurrent Layer')
        axs[0, 2].set_xlim([0, max_ro])
        axs[0, 2].set_title('Recoding')
        return domain

    def animate(t):
        domain = clear()

        u = np.round(l * mu[t], 2)
        axs[0, 4].text(-.2, .47, r'$u_t = L \; \frac{\bf{b} \cdot \bf{r}}{\bf{a} \cdot \bf{r}} = $' + str(u),
                       fontsize=15, style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

        # random walk
        world_ax.plot(steps[:t + 1], x[:t + 1], 'g')
        world_ax.scatter(t, x[t], s=dot_size, c='green')

        # Likelihood
        world_ax.plot(steps[:t + 1], y[:t + 1], 'r.')
        pdf_likelihood = norm.pdf(domain, y[t], sqrt_r)
        world_ax.fill_betweenx(domain, t + pdf_likelihood * (T_max * pad), t, color='crimson', alpha=0.5,
                               label='likelihood', edgecolor="crimson", linewidth=0)
        world_ax.plot(t + pdf_likelihood * (T_max * pad), domain, color='crimson', linewidth=2.0)
        world_ax.scatter(t, y[t], s=dot_size, c='crimson')

        # Prior
        pdf_prior = norm.pdf(domain, mu[t - 1], sqrt_sig[t - 1])
        world_ax.fill_betweenx(domain, t + pdf_prior * (T_max * pad), t, color='dodgerblue', alpha=0.5, label='prior',
                               edgecolor="dodgerblue", linewidth=0)
        world_ax.plot(t + pdf_prior * (T_max * pad), domain, color='dodgerblue', linewidth=2.0)

        # Posterior
        world_ax.plot(steps[:t + 1], mu[:t + 1], 'k')
        world_ax.fill_between(steps[:t + 1], mu[:t + 1] + sqrt_sig[:t + 1], mu[:t + 1] - sqrt_sig[:t + 1], color='gray',
                              alpha=0.5)
        pdf_post = norm.pdf(domain, mu[t], sqrt_sig[t])
        world_ax.fill_betweenx(domain, t + pdf_post * (T_max * pad), t, color='black', alpha=0.5, label='posterior',
                               edgecolor="black", linewidth=0)
        world_ax.plot(t + pdf_post * (T_max * pad), domain, color='black', linewidth=2.0)
        world_ax.scatter(t, mu[t], s=dot_size, c='black')

        # neural activity @ input layer
        axs[0, 0].barh(neurons_in, ri[t], color=custom_cmap_i(rescale_i(neurons_in)))

        # neural activity @ rec layer
        axs[0, 2].barh(neurons_rec, ro[t], color=custom_cmap_o(rescale_o(neurons_rec)))
        world_ax.legend(ncol=3, loc='upper left')

    clear()
    anim = FuncAnimation(
        fig,
        animate,
        frames=T-1,
        interval=1,
        blit=False,
        repeat=False
    )

    plt.show()

    return anim

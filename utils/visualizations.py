import torch
from sklearn.metrics import confusion_matrix
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os
from typing import Optional, List


def plot_attention_maps(input_data, attn_maps, idx=0) -> None:
    """
    Adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """
    """Plot the attention map

    Args:
        input_data: should be None or [batch_size, seq_len]
        attn_maps: should have size [num_laer, batch_size, num_head, seq_len, model_dim]
        idx: batch index (which sample of a batch)

    """
    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])
    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]

    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 4 if num_heads == 1 else 3
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads * fig_size, num_layers * fig_size))
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    for row in range(num_layers):
        for column in range(num_heads):
            ax[row][column].imshow(attn_maps[row][column], origin='lower', vmin=0)
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_xticklabels(input_data.tolist())
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_yticklabels(input_data.tolist())
            ax[row][column].set_title(f"Layer {row + 1}, Head {column + 1}")
    fig.subplots_adjust(hspace=0.5)
    plt.show()


def plot_confusion_matrix(outs: np.ndarray, labels: np.ndarray, save_pth: Optional[str] = None):
    """plot and save the confusion matrix
    Args:
        outs - the output of inference
        labels - the ground truth
        save_pth - where to save the confusion matrix pic
    """
    accu = (outs == labels).mean()
    conf_mtr = confusion_matrix(outs, labels)
    plt.figure(figsize=(10, 10))
    sb.set(font_scale=1.2)
    sb.heatmap(conf_mtr, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.ylabel('Predicted Labels')
    plt.xlabel('True Labels')
    plt.title(f'Confusion Matrix(Accuracy:{int(accu * 100) / 100})')
    plt.xticks(np.arange(10), labels=range(10), rotation=0)
    plt.yticks(np.arange(10), labels=range(10), rotation=0)
    if save_pth is not None:
        plt.savefig(os.path.join(save_pth, 'confusion_matrix.png'), bbox_inches='tight')
    plt.show()


def scatter_tsne(data: list, mods: list, names: list, out_path: Optional[str] = None):
    """
    plot 2d tsne scatter figure (warning: has to be more than 1 trajectory or 'axes' object is not subscriptable)
    Args:
        data: list of numpy array [traj 1 (e.g. if s steps x m mods:(s, m, D)), traj 2, ... , traj N]
        mods: list of modalities [name of mod1, name of mod2, ..., name of mod m]
        names: list of str or int value [name1, name2, ... , nameN] (indicates which traj)
        out_path: where to save the output figure
    Returns:

    """
    shape_list = ['o', 'v', 's', '*', 'd', 'p', 'P', 'x', '1', '+']
    mod_shape_dict = {mod: shape_list[idx] for idx, mod in enumerate(mods)}
    assert len(data) == len(names)
    assert data[0].shape[1] == len(mods)
    from sklearn.manifold import TSNE
    from matplotlib.colors import Normalize
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    num_traj = len(names)
    fig, ax = plt.subplots(num_traj, 1, figsize=(1 * 20, num_traj * 10))
    for traj_idx, (x, name) in enumerate(zip(data, names)):
        # color = np.arange(x.shape[1])
        # norm = Normalize(vmin=0, vmax=1)
        # cmap = plt.cm.RdYlBu
        # color = cmap(norm(color))
        num_steps = x.shape[0]
        num_mod = x.shape[1]
        x_tsne = tsne.fit_transform(x.reshape(-1, x.shape[-1]))
        print(f"t-SNE KL-Divergence of {name}: {tsne.kl_divergence_}")
        x_tsne = x_tsne.reshape(num_steps, num_mod, -1)
        x_tsne = np.transpose(x_tsne, (1, 0, 2))

        for i, mod in enumerate(mods):
            x_ = x_tsne[i, :]
            scatter = ax[traj_idx].scatter(x_[:, 0], x_[:, 1], marker=mod_shape_dict[mod], label=mod,
                                           c=np.arange(num_steps), cmap='viridis', edgecolor='k', s=32)
        legend = ax[traj_idx].legend(*scatter.legend_elements(), title='progress', bbox_to_anchor=(1.06, 1.0))
        ax[traj_idx].add_artist(legend)
        ax[traj_idx].legend()
        ax[traj_idx].set_title(f'Trajectory {int(name.item()) if isinstance(name, torch.Tensor) else name}')
        ax[traj_idx].set_xlabel('x')
        ax[traj_idx].set_ylabel('y')

    # plt.text(0, 0, '▲', fontsize=12, verticalalignment='center', horizontalalignment='center')
    # # plt.text(x_position, y_position, '■', fontsize=12, verticalalignment='center', horizontalalignment='center')

    fig.suptitle('t-SNE Visualization')
    if out_path is not None:
        fig.savefig(out_path + '_tsne.png', dpi=300)
    # plt.show()
    plt.clf()
    plt.close('all')


def scatter_tsne_selected(data: List[np.ndarray], mods: List[str], names: List[torch.tensor],
                          selected_name: List[int],
                          out_path: Optional[str] = None):
    """

    Args:
        data: List of numpy array [traj 1 (e.g. if s steps x m mods:(s, m, D)), traj 2, ... , traj N]
        mods: List of modalities [name of mod1, name of mod2, ..., name of mod m]
        names: List of str or int value [name1, name2, ... , nameN] (indicates which traj)
        selected_name: List of trajectory (names) to be visualized
        out_path: where to save the output figure
    Returns:

    """

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from matplotlib.colors import Normalize
    from matplotlib.cm import get_cmap

    shape_list = ['o', 'v', 's', '*', 'd', 'p', 'P', 'x', '1', '+']
    mod_shape_dict = {mod: shape_list[idx] for idx, mod in enumerate(mods)}
    color_maps = [
        'Greys', 'Reds', 'Blues', 'Greens', 'Oranges', 'Purples',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    plot_color_maps = ['k', 'r', 'b', 'g', 'y', ]
    assert len(data) == len(names)
    assert data[0].shape[1] == len(mods)

    # Concatenate all data points of selected trajectories
    selected_trajs = []
    selected_traj_start_end = []
    end = 0
    for select_name in selected_name:
        traj = data[names.index(select_name)]
        selected_trajs.append(traj)
        start = end
        end += traj.shape[0]
        selected_traj_start_end.append([start, end])
    selected_trajs = np.concatenate(selected_trajs, axis=0)

    # Create a color map for selected trajectories
    # cmap = get_cmap('tab10')
    # colors = cmap(np.arange(len(selected_name)))

    # Apply t-SNE on the concatenated data
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    num_steps = selected_trajs.shape[0]
    num_mod = selected_trajs.shape[1]
    x_tsne = tsne.fit_transform(selected_trajs.reshape(-1, selected_trajs.shape[-1]))
    print(tsne.kl_divergence_)
    x_tsne = x_tsne.reshape(num_steps, num_mod, -1)
    selected_trajs = [np.transpose(x_tsne[start:end], (1, 0, 2)) for start, end in selected_traj_start_end]

    num_selected_trajectories = len(selected_name)
    fig, ax = plt.subplots(num_selected_trajectories + 1, 2, figsize=(2 * 10, (num_selected_trajectories + 1) * 10))

    for traj_idx, (x_tsne, traj) in enumerate(zip(selected_trajs, selected_name)):
        for i, mod in enumerate(mods):
            x_ = x_tsne[i, :]
            brightness = 0.999 * np.arange(x_.shape[0]) / x_.shape[
                0] + 0.001  # Linear mapping for brightness to [0.2, 0.8] to avoid white points

            # scatter.set_facecolor(cmap(brightness))
            if i == 0:
                scatter = ax[traj_idx][1].plot(x_[:, 0], x_[:, 1], marker=mod_shape_dict[mod], label=mod,
                                               c=plot_color_maps[traj_idx],
                                               alpha=1.0, linewidth=0.5)
                scatter_all = ax[-1][1].plot(x_[:, 0], x_[:, 1], marker=mod_shape_dict[mod],
                                             label=f'Trajectory{traj}:{mod}',
                                             c=plot_color_maps[traj_idx],
                                             alpha=1.0, linewidth=0.5, )
                scatter = ax[traj_idx][0].plot(x_[:, 0], x_[:, 1],
                                               c=plot_color_maps[traj_idx],
                                               alpha=0.3, linewidth=0.5)

            scatter = ax[traj_idx][0].scatter(x_[:, 0], x_[:, 1], marker=mod_shape_dict[mod], label=mod,
                                              c=brightness, cmap=color_maps[traj_idx],
                                              edgecolor='k', alpha=1.0, linewidth=0.5, s=30)

            scatter_all = ax[-1][0].scatter(x_[:, 0], x_[:, 1], marker=mod_shape_dict[mod],
                                            label=f'Trajectory{traj}:{mod}',
                                            c=brightness, cmap=color_maps[traj_idx],
                                            edgecolor='k', alpha=1.0, linewidth=0.5, s=30, vmin=-0.6, vmax=1.0)

        # legend = ax[traj_idx][0].legend(*scatter.legend_elements(), title='progress', bbox_to_anchor=(1.06, 1.0))
        # ax[traj_idx][0].add_artist(legend)
        # ax[traj_idx][0].legend()
        ax[traj_idx][0].set_title(f'Trajectory {traj}')
        ax[traj_idx][0].set_xlabel('x')
        ax[traj_idx][0].set_ylabel('y')

    # ax[-1].legend()
    ax[-1][0].set_title('All Selected Trajectories')
    ax[-1][0].set_xlabel('x')
    ax[-1][0].set_ylabel('y')

    # fig.suptitle('t-SNE Visualization for Selected Trajectories')

    if out_path is not None:
        fig.savefig(out_path + '_tsne_selected.png', dpi=300)

    # plt.show()
    plt.clf()
    plt.close('all')


# Example usage:
# scatter_tsne_selected(data=[traj1, traj2, traj3], mods=['mod1', 'mod2'], names=[14, 15, 16, 30, 31, 32], selected_trajectories=[14, 15, 16, 30, 31, 32], out_path='output_path')


def scatter_tnse_3d(data: list, mods: list, names: list, out_path: Optional[str] = None):
    """

    Args:
        data: list of numpy array [traj 1 (e.g. if s steps x m mods:(s, m, D)), traj 2, ... , traj N]
        mods: list of modalities [name of mod1, name of mod2, ..., name of mod m]
        names: list of str or int value [name1, name2, ... , nameN] (indicates which traj)
        out_path: where to save the output figure
    Returns:

    """
    shape_list = ['o', 'v', 's', '*', 'd', 'p', 'P', 'x', '1', '+']
    mod_shape_dict = {mod: shape_list[idx] for idx, mod in enumerate(mods)}
    assert len(data) == len(names)
    assert data[0].shape[1] == len(mods)
    from sklearn.manifold import TSNE
    from matplotlib.colors import Normalize
    tsne = TSNE(n_components=3, random_state=42, perplexity=5)
    num_traj = len(names)
    fig, ax = plt.subplots(num_traj, 1, figsize=(1 * 10, num_traj * 5), subplot_kw=dict(projection='3d'))
    for traj_idx, (x, name) in enumerate(zip(data, names)):
        # color = np.arange(x.shape[1])
        # norm = Normalize(vmin=0, vmax=1)
        # cmap = plt.cm.RdYlBu
        # color = cmap(norm(color))
        num_steps = x.shape[0]
        num_mod = x.shape[1]
        x_tsne = tsne.fit_transform(x.reshape(-1, x.shape[-1]))
        x_tsne = x_tsne.reshape(num_steps, num_mod, -1)
        x_tsne = np.transpose(x_tsne, (1, 0, 2))

        for i, mod in enumerate(mods):
            x_ = x_tsne[i, :]
            scatter = ax[traj_idx].scatter(x_[:, 0], x_[:, 1], x_[:, 2], marker=mod_shape_dict[mod], label=mod,
                                           c=np.arange(num_steps), cmap='viridis', edgecolor='k')
        legend = ax[traj_idx].legend(*scatter.legend_elements(), title='progress', bbox_to_anchor=(1.3, 1.0))
        ax[traj_idx].add_artist(legend)
        ax[traj_idx].legend(bbox_to_anchor=(1.1, 1.0))
        ax[traj_idx].set_title(f'Trajectory {int(name.item())}')
        ax[traj_idx].set_xlabel('x')
        ax[traj_idx].set_ylabel('y')

    # plt.text(0, 0, '▲', fontsize=12, verticalalignment='center', horizontalalignment='center')
    # # plt.text(x_position, y_position, '■', fontsize=12, verticalalignment='center', horizontalalignment='center')

    fig.suptitle('T-SNE Visualization')
    if out_path is not None:
        fig.savefig(out_path + '_tsne_3d.png', dpi=300)
    plt.show()
    plt.clf()
    plt.close('all')


def scatter_pca(data: list, mods: list, names: list, out_path: Optional[str] = None):
    """

    Args:
        data: list of numpy array [traj 1 (e.g. if s steps x m mods:(s, m, D)), traj 2, ... , traj N]
        mods: list of modalities [name of mod1, name of mod2, ..., name of mod m]
        names: list of str or int value [name1, name2, ... , nameN] (indicates which traj)
        out_path: where to save the output figure
    Returns:

    """
    shape_list = ['o', 'v', 's', '*', 'd', 'p', 'P', 'x', '1', '+']
    mod_shape_dict = {mod: shape_list[idx] for idx, mod in enumerate(mods)}
    assert len(data) == len(names)
    assert data[0].shape[1] == len(mods)
    from sklearn.decomposition import PCA
    from matplotlib.colors import Normalize
    pca = PCA(n_components=2)
    num_traj = len(names)
    fig, ax = plt.subplots(num_traj, 1, figsize=(1 * 10, num_traj * 5))
    for traj_idx, (x, name) in enumerate(zip(data, names)):
        # color = np.arange(x.shape[1])
        # norm = Normalize(vmin=0, vmax=1)
        # cmap = plt.cm.RdYlBu
        # color = cmap(norm(color))
        num_steps = x.shape[0]
        num_mod = x.shape[1]
        x_tsne = pca.fit_transform(x.reshape(-1, x.shape[-1]))
        x_tsne = x_tsne.reshape(num_steps, num_mod, -1)
        x_tsne = np.transpose(x_tsne, (1, 0, 2))

        for i, mod in enumerate(mods):
            x_ = x_tsne[i, :]
            scatter = ax[traj_idx].scatter(x_[:, 0], x_[:, 1], marker=mod_shape_dict[mod], label=mod,
                                           c=np.arange(num_steps), cmap='viridis', edgecolor='k')
        legend = ax[traj_idx].legend(*scatter.legend_elements(), title='progress', bbox_to_anchor=(1.06, 1.0))
        ax[traj_idx].add_artist(legend)
        ax[traj_idx].legend()
        ax[traj_idx].set_title(f'Trajectory {int(name.item())}')
        ax[traj_idx].set_xlabel('x')
        ax[traj_idx].set_ylabel('y')

    # plt.text(0, 0, '▲', fontsize=12, verticalalignment='center', horizontalalignment='center')
    # # plt.text(x_position, y_position, '■', fontsize=12, verticalalignment='center', horizontalalignment='center')

    fig.suptitle('PCA Visualization')
    if out_path is not None:
        fig.savefig(out_path + '_pca.png', dpi=300)
    plt.show()
    plt.clf()
    plt.close('all')


def scatter_pca_3d(data: list, mods: list, names: list, out_path: Optional[str] = None):
    """

    Args:
        data: list of numpy array [traj 1 (e.g. if s steps x m mods:(s, m, D)), traj 2, ... , traj N]
        mods: list of modalities [name of mod1, name of mod2, ..., name of mod m]
        names: list of str or int value [name1, name2, ... , nameN] (indicates which traj)
        out_path: where to save the output figure
    Returns:

    """
    shape_list = ['o', 'v', 's', '*', 'd', 'p', 'P', 'x', '1', '+']
    mod_shape_dict = {mod: shape_list[idx] for idx, mod in enumerate(mods)}
    assert len(data) == len(names)
    assert data[0].shape[1] == len(mods)
    from sklearn.decomposition import PCA
    from matplotlib.colors import Normalize
    pca = PCA(n_components=3)
    num_traj = len(names)
    fig, ax = plt.subplots(num_traj, 1, figsize=(1 * 10, num_traj * 5), subplot_kw=dict(projection='3d'))
    for traj_idx, (x, name) in enumerate(zip(data, names)):
        # color = np.arange(x.shape[1])
        # norm = Normalize(vmin=0, vmax=1)
        # cmap = plt.cm.RdYlBu
        # color = cmap(norm(color))
        num_steps = x.shape[0]
        num_mod = x.shape[1]
        x_tsne = pca.fit_transform(x.reshape(-1, x.shape[-1]))
        x_tsne = x_tsne.reshape(num_steps, num_mod, -1)
        x_tsne = np.transpose(x_tsne, (1, 0, 2))

        for i, mod in enumerate(mods):
            x_ = x_tsne[i, :]
            scatter = ax[traj_idx].scatter(x_[:, 0], x_[:, 1], x_[:, 2], marker=mod_shape_dict[mod], label=mod,
                                           c=np.arange(num_steps), cmap='viridis', edgecolor='k')
        legend = ax[traj_idx].legend(*scatter.legend_elements(), title='progress', bbox_to_anchor=(1.3, 1.0))
        ax[traj_idx].add_artist(legend)
        ax[traj_idx].legend(bbox_to_anchor=(1.1, 1.0))
        ax[traj_idx].set_title(f'Trajectory {int(name.item())}')
        ax[traj_idx].set_xlabel('x')
        ax[traj_idx].set_ylabel('y')

    fig.suptitle('PCA Visualization')
    if out_path is not None:
        fig.savefig(out_path + '_pca_3d.png', dpi=300)
    plt.show()
    plt.clf()
    plt.close('all')


if __name__ == '__main__':
    data = [np.random.rand(10, 3, 512), np.random.rand(7, 3, 512), np.random.rand(12, 3, 512),
            np.random.rand(13, 3, 512)]
    mods = ['a', 'b', 'c']
    names = [i for i in range(4)]

    scatter_tsne_selected(data, mods, names, [0, 1, 2])

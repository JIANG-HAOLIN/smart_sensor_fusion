from sklearn.metrics import confusion_matrix
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os
from typing import Optional


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

    Args:
        data: list of numpy array [traj 1 (e.g. if s steps x m mods:(m, s, D)), traj 2, ... , traj N]
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
    fig, ax = plt.subplots(num_traj, 1, figsize=(1 * 10, num_traj * 5))
    for traj_idx, (x, name) in enumerate(zip(data, names)):
        # color = np.arange(x.shape[1])
        # norm = Normalize(vmin=0, vmax=1)
        # cmap = plt.cm.RdYlBu
        # color = cmap(norm(color))
        num_steps = x.shape[0]
        num_mod = x.shape[1]
        x_tsne = tsne.fit_transform(x.reshape(-1, x.shape[-1]))
        print(tsne.kl_divergence_)
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

    fig.suptitle('t-SNE Visualization')
    if out_path is not None:
        fig.savefig(out_path, dpi=300)
    plt.show()
    plt.clf()
    plt.close('all')

# data = [np.random.rand(10,3,512), np.random.rand(7,3,512)]
# mods = ['a', 'b', 'c']
# names = [str(i) for i in range(2)]
#
# scatter_tsne(data, mods, names)

from textwrap import wrap

from matplotlib import pyplot as plt


def stack_images(image_list, path, text_list=None, title=None):
    # stack multiple images into one figure
    ncols = 10
    fig, ax = plt.subplots(
        nrows=len(image_list) // ncols + 1,
        ncols=ncols,
        squeeze=False,
        figsize=((ncols * 5, (len(image_list) // ncols + 1) * 5)),
    )
    for iter_idx in range(len(ax.flatten())):
        if iter_idx >= len(image_list):
            ax[iter_idx // ncols, iter_idx % ncols].axis("off")
            continue
        ax[iter_idx // ncols, iter_idx % ncols].axis("off")
        ax[iter_idx // ncols, iter_idx % ncols].imshow(image_list[iter_idx])
        if text_list is not None:
            ax[iter_idx // ncols, iter_idx % ncols].set_title(
                text_list[iter_idx], rotation=20, wrap=True
            )
            # ax[iter_idx // ncols, iter_idx % ncols].set_title(
            #     "\n".join(
            #         wrap(
            #             text_list[iter_idx],
            #             60,
            #         )
            #     )
            # )

    # for iter_idx, image in enumerate(image_list):
    #     ax[iter_idx // ncols, iter_idx % ncols].axis("off")
    #     ax[iter_idx // ncols, iter_idx % ncols].imshow(image)
    fig.suptitle(title)
    plt.savefig(path)
    plt.close()

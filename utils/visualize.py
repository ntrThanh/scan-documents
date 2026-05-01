import numpy as np
import matplotlib.pyplot as plt

def show(images, titles=None, cmap=None, figsize=None, cols=3):
    if not isinstance(images, (list, tuple)):
        images = [images]
        titles = [titles] if titles is not None else [None]
    else:
        if titles is None:
            titles = [None] * len(images)
        elif not isinstance(titles, (list, tuple)):
            titles = [titles] * len(images)

    n = len(images)
    cols = min(n, cols)
    rows = (n + cols - 1) // cols

    if figsize is None:
        figsize = (5 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, (img, title) in enumerate(zip(images, titles)):
        ax = axes[i]
        ax.axis("off")

        if img is None:
            continue

        if img.ndim == 2:
            ax.imshow(img, cmap=cmap or "gray")
        else:
            ax.imshow(img)

        if title:
            ax.set_title(title)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
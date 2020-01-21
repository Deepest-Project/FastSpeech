from text import *
import matplotlib.pyplot as plt


def plot_image(target, melspec, alignments, text, mel_lengths, text_lengths):
    fig, axes = plt.subplots(3,1,figsize=(20,20))
    L, T = text_lengths[-1], mel_lengths[-1]

    axes[0].imshow(target[-1].detach().cpu()[:,:T],
                   origin='lower',
                   aspect='auto')

    axes[1].imshow(melspec[-1].detach().cpu()[:,:T],
                   origin='lower',
                   aspect='auto')

    axes[2].imshow(alignments[-1].transpose(0,1).detach().cpu()[:L,:T],
                   origin='lower',
                   aspect='auto')

    plt.xticks(range(T), [ f'{i}' if (i%10==0 or i==T-1) else ''
           for i in range(T) ])

    plt.yticks(range(L),
           sequence_to_text(text[-1].detach().cpu().numpy()[:L]))
    
    return fig
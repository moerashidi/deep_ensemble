import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def calc_bins(probs, preds, labels, num_bins=10):
    bins = np.linspace(0.1, 1.01, num_bins)
    binned = np.digitize(probs, bins)

    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(probs[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels[binned==bin]==preds[binned==bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (probs[binned==bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(probs, preds, labels, num_sample):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(probs, preds, labels, num_sample)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE

def draw_reliability_graph(probs, preds, labels, num_sample=10):
    ECE, MCE = get_metrics(probs, preds, labels, num_sample)
    bins, _, bin_accs, _, _ = calc_bins(probs, preds, labels, num_sample)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)

    plt.xlabel('Predicted Confidence')
    plt.ylabel('True Confidence')

    ax.set_axisbelow(True) 

    plt.plot(bins[bin_accs!=0], bin_accs[bin_accs!=0], marker='o', c='r')

    plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

    plt.gca().set_aspect('equal', adjustable='box')

    ECE_patch = mpatches.Patch(
        color='b',
        label='Expected Calibration Error = {:.2f}%'.format(ECE*100)
    )
    MCE_patch = mpatches.Patch(
        color='k', 
        label='Max Calibration Error = {:.2f}%'.format(MCE*100)
    )
    plt.legend(handles=[ECE_patch, MCE_patch])

    plt.show()
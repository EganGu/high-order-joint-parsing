import matplotlib.pyplot as plt
import numpy as np


class Scaler(object):

    def __init__(self, base=1., transform=None, eps=1e-12, s_max=None, s_min=None):
        self.max = s_max
        self.min = s_min
        self.base = base
        self.transform = transform
        self.eps = eps

    def map(self, x, shift=0):
        if self.transform == 'log':
            x = np.log([i+self.eps for i in x])
        elif self.transform == 'square':
            x = np.square([i for i in x])
        if self.max is None:
            self.max = max(x)
        if self.min is None:
            self.min = min(x)
        return [((i-self.min) * self.base / (self.max-self.min)) + shift for i in x]

    def inv(self, x):
        assert self.max is not None and self.min is not None
        x = [i * (self.max-self.min) / self.base + self.min for i in x]
        if self.transform == 'log':
            x = np.exp(x)
        elif self.transform == 'square':
            x = np.sqrt([i for i in x])
        return x


def AdaptiveBinning(infer_results, show_reliability_diagram=False, m_type='squence_acc', rate=0.25, fname='show.png'):
    '''
    This function implement adaptive binning. It returns AECE, AMCE and some other useful values.

    Arguements:
    infer_results (list of list): a list where each element "res" is a two-element list denoting the infer result of a single sample.
    res[0] is the confidence score r and res[1] is the correctness score c. Since c is either 1 or 0, here res[1] if the prediction is correctd and False otherwise.
    show_reliability_diagram (boolean): a boolean value to denote wheather to plot a Reliability Diagram.

    Return Values:
    AECE (float): expected calibration error based on adaptive binning.
    AMCE (float): maximum calibration error based on adaptive binning.
    cofidence (list): average confidence in each bin.
    accuracy (list): average accuracy in each bin.
    cof_min (list): minimum of confidence in each bin.
    cof_max (list): maximum of confidence in each bin.

    '''
    MAX_COF = 9999
    # Intialize.
    infer_results.sort(key=lambda x: x[0], reverse=True)
    n_total_sample = len(infer_results)

    # assert infer_results[0][0] <= 1 and infer_results[1][0] >= 0, 'Confidence score should be in [0,1]'

    z = 1.645
    num = [0 for i in range(n_total_sample)]
    final_num = [0 for i in range(n_total_sample)]
    confidence = [0 for i in range(n_total_sample)]
    cof_min = [MAX_COF for i in range(n_total_sample)]
    cof_max = [0 for i in range(n_total_sample)]
    metric = [0 for i in range(n_total_sample)]

    ind = 0
    target_number_samples = float('inf')

    raw_x = [x[0] for x in infer_results]
    confidence_score_set = Scaler(1.).map(raw_x)
    # Traverse all samples for a initial binning.
    for i, confidence_score in enumerate(confidence_score_set):
        # Merge the last bin if too small.
        if num[ind] > target_number_samples:
            if (n_total_sample - i) > 40:
                # and cof_min[ind] - infer_results[-1][0] > 0.05
                ind += 1
                target_number_samples = float('inf')
        num[ind] += 1
        confidence[ind] += confidence_score

        cof_min[ind] = min(cof_min[ind], confidence_score)
        cof_max[ind] = max(cof_max[ind], confidence_score)
        # Get target number of samples in the bin.
        if cof_max[ind] == cof_min[ind]:
            target_number_samples = float('inf')
        else:
            target_number_samples = (
                z / (cof_max[ind] - cof_min[ind])) ** 2 * rate
    n_bins = ind + 1

    # Get final binning.
    # if target_number_samples - num[ind] > 0:
    #     # less than target num
    #     needed = target_number_samples - num[ind]
    #     # extract = [0 for i in range(n_bins - 1)]
    #     extract = int(needed * num[ind] / n_total_sample)
    #     final_num[n_bins - 1] = num[n_bins - 1]
    #     for i in range(n_bins - 1):
    #         # extract[i] = int(needed * num[ind] / n_total_sample)
    #         final_num[i] = num[i] - extract
    #         final_num[n_bins - 1] += extract
    # else:
    final_num = num
    final_num = final_num[:n_bins]

    # Re-intialize.
    num = [0 for i in range(n_bins)]
    confidence = [0 for i in range(n_bins)]
    cof_min = [MAX_COF for i in range(n_bins)]
    cof_max = [0 for i in range(n_bins)]
    metric = [0 for i in range(n_bins)]
    # gap = [0 for i in range(n_bins)]
    # neg_gap = [0 for i in range(n_bins)]
    # Bar location and width.
    x_location = [0 for i in range(n_bins)]
    width = [0 for i in range(n_bins)]
    if m_type == 'nested_acc':
        correct = [[0, 0] for i in range(n_bins)]
    else:
        correct = [0 for i in range(n_bins)]

    # Calculate confidence and accuracy in each bin.
    ind = 0
    for i, confindence_correctness in enumerate(infer_results):
        confidence_score = confindence_correctness[0]
        correctness = confindence_correctness[1]
        num[ind] += 1
        confidence[ind] += confidence_score
        if m_type == 'squence_acc':
            correct[ind] += correctness
        elif m_type == 'nested_acc':
            correct[ind][0] += correctness[0]
            correct[ind][1] += correctness[1]
        elif m_type == 'acc' and correctness:
            correct[ind] += 1
        cof_min[ind] = min(cof_min[ind], confidence_score)
        cof_max[ind] = max(cof_max[ind], confidence_score)

        if num[ind] == final_num[ind]:
            if m_type == 'squence_acc':
                metric[ind] = correct[ind] / \
                    confidence[ind] if confidence[ind] > 0 else 0
            elif m_type == 'nested_acc':
                metric[ind] = correct[ind][0] / \
                    correct[ind][1] if correct[ind][1] > 0 else 0
            elif m_type == 'acc':
                metric[ind] = correct[ind] / num[ind] if num[ind] > 0 else 0
            confidence[ind] = confidence[ind] / num[ind] if num[ind] > 0 else 0
            left = cof_min[ind]
            right = cof_max[ind]
            x_location[ind] = (left + right) / 2
            width[ind] = (right - left) * 0.9
            # if confidence[ind] - accuracy[ind] > 0:
            #     gap[ind] = confidence[ind] - accuracy[ind]
            # else:
            #     neg_gap[ind] = confidence[ind] - accuracy[ind]
            ind += 1

    # Get AECE and AMCE based on the binning.
    # AMCE = 0
    # AECE = 0
    # for i in range(n_bins):
    #     AECE += abs((accuracy[i] - confidence[i])) * \
    #         final_num[i] / n_total_sample
    #     AMCE = max(AMCE, abs((accuracy[i] - confidence[i])))

    # Plot the Reliability Diagram if needed.
    if show_reliability_diagram:
        # f1, ax = plt.subplots()
        # plt.bar(x_location, accuracy, width)
        plt.scatter(x_location, metric, s=Scaler(
            100, transform='log').map(final_num))
        # .scatter
        # plt.bar(x_location, gap, width, bottom=accuracy)
        # plt.bar(x_location, neg_gap, width, bottom=accuracy)
        # plt.legend(['Accuracy', 'Positive gap', 'Negative gap'],
        #            fontsize=18, loc=2)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        plt.xlabel('Confidence', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)
        plt.savefig(fname, dpi=300)
        # plt.show()

    # return AECE, AMCE, cof_min, cof_max, confidence, accuracy
    return x_location, metric, final_num

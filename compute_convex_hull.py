# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Computes the convex hull of Precision-Recall/ROC points returned by sklearn. """


def _compute_stepwise_convex_hull_precision_recall(recall, precision):
    """ Computes the step-wise convex hull for precision-recall points. """
    # for same recall, keep maximum precision
    recall_points = {}
    for r, p in sorted(zip(recall, precision)):
        if r not in recall_points or p > recall_points[r]:
            recall_points[r] = p

    # for same precision, keep maximum recall
    points = [(0, 1)]
    cur_r = 0
    for r, p in sorted(recall_points.items(), key=lambda x: (x[1], x[0]), reverse=True):
        if cur_r < r:
            points.append((r, p))
            cur_r = r
    if (1, 0) not in points:
        points.append((1, 0))

    # create curve points
    curve_points = []
    last_r, last_p = 0, 1
    for r, p in points:
        if len(curve_points) > 0 and last_p > p:
            curve_points.append((last_r, p))
        curve_points.append((r, p))
        last_r = r

    return zip(*curve_points)


def compute_stepwise_convex_hull(x, y, mode='roc'):
    if mode == 'roc':
        x_selected, y_selected = _compute_stepwise_convex_hull_precision_recall([-p + 1 for p in x], y)
        return [-x + 1 for x in x_selected], y_selected
    else:
        return _compute_stepwise_convex_hull_precision_recall(x, y)

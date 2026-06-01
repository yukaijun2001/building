import os

import numpy as np


def save_npy_results(setting, preds, trues, metrics, result_root=None):
    if result_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result_root = os.path.join(project_root, "result")

    folder_path = os.path.join(result_root, setting)
    os.makedirs(folder_path, exist_ok=True)

    np.save(os.path.join(folder_path, "metrics.npy"), np.asarray(metrics))
    np.save(os.path.join(folder_path, "pred.npy"), np.asarray(preds))
    np.save(os.path.join(folder_path, "true.npy"), np.asarray(trues))

    print("numpy result saved to: {}".format(folder_path))
    return folder_path

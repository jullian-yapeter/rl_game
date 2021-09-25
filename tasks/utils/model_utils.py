import numpy as np
import torch as T


def save_checkpoint(model, task_name=None):
    if task_name:
        T.save(model.state_dict(), f"{model.checkpoint_file}_{task_name}")
    else:
        T.save(model.state_dict(), model.checkpoint_file)


def load_checkpoint(model, task_name=None, checkpoint_file=None):
    if checkpoint_file:
        model.load_state_dict(T.load(checkpoint_file, map_location=model.device))
    elif task_name:
        model.load_state_dict(T.load(f"{model.checkpoint_file}_{task_name}", map_location=model.device))
    else:
        model.load_state_dict(T.load(model.checkpoint_file, map_location=model.device))


def shuffled_indices(num_indicies):
    indices = np.arange(num_indicies, dtype=np.int64)
    np.random.shuffle(indices)
    return indices

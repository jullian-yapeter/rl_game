import torch as T


def save_checkpoint(model, checkpoint_file=None):
    if checkpoint_file:
        T.save(model.state_dict(), checkpoint_file)
    T.save(model.state_dict(), model.checkpoint_file)


def load_checkpoint(model, checkpoint_file=None):
    if checkpoint_file:
        model.load_state_dict(T.load(model.checkpoint_file, map_location=model.device))
    model.load_state_dict(T.load(model.checkpoint_file, map_location=model.device))

""" Util class for saving and loading models """
import pickle

class ModelSaver(object):
    """Class to save and load model ckpts."""

    def __init__(self, save_dir):
        """Initialize saver/loader."""
        self.model_save_path = save_dir / "model.pkl"

    def save(self, model):
        """Save the model."""
        print(f"Saving model to {self.model_save_path}.")
        with self.model_save_path.open("wb") as f:
            pickle.dump(model, f)

    def load(self):
        """Load the model."""
        with self.model_save_path.open("rb") as f:
            model = pickle.load(f)

        return model
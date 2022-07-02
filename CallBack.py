import utils
import tensorflow as tf
from tensorflow import keras

class callback(keras.callbacks.Callback):

    def __init__(self,dataset):
        super().__init__()
        self.dataset = dataset
        self.idx_mapping = utils.idx_mapping()

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_logits = model.predict(X)
            decoded_labels = utils.GreedyDecoder(batch_logits)
            batch_predictions = utils.get_batch_transcripts(decoded_labels)
            batch_transcripts = ["".join([self.idx_mapping[i] for i in label]) for label in y]
            
            predictions.extend(batch_predictions)
            targets.extend(batch_transcripts)
            wer_score = wer(targets, predictions)
        print("\n")
        print(f"WER SCORE: {wer_score}")
        print("*"*15)
        for i in np.random.randint(0, 32, 2):
            print(f"Transcript: {targets[i]}")
            print(f"prediction: {predictions[i]}")
            print("*"*15)
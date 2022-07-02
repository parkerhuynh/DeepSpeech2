from typing import Callable
import tensorflow as tf

def CTC_loss() -> Callable:
        def get_length(tensor):
            lengths = tf.math.reduce_sum(tf.ones_like(tensor), 1)
            return tf.cast(lengths, tf.int32)

        def ctc_loss(labels, logits):
            label_length = get_length(labels)
            logit_length = get_length(tf.math.reduce_max(logits, 2))
            labels = tf.cast(labels, tf.int32)
            return tf.nn.ctc_loss(labels, logits, label_length, logit_length,
                                  logits_time_major=False, blank_index=-1)
        return ctc_loss

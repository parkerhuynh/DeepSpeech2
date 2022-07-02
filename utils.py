import numpy as np
import itertools
from jiwer import wer
from scipy.io import wavfile
from IPython import display
import matplotlib.pyplot as plt
import numpy as np

def chr_mapping():
    character_map = {' ': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
                'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20,
                'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, "'": 27,'': 28}
    return character_map

def idx_mapping():
    idx_mapping = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l','m',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'", '']
    return idx_mapping

def GreedyDecoder(batch_logits):
    best_candidates = np.argmax(batch_logits, axis=2)
    decoded = [np.array([k for k, _ in itertools.groupby(best_candidate)]) for best_candidate in best_candidates]
    return decoded

def get_batch_transcripts(sequences):
    idx_mapping = idx_mapping()
    return [''.join(idx_mapping[char_label] for char_label in sequence if char_label not in (-1, 28)) for sequence in sequences]

def evaluate(model, dataset):
    loss = model.evaluate(dataset)
    predictions = []
    targets = []
    batch_num = 1
    mean_wer = 0
    for batch in dataset:
        X, y = batch
        batch_logits = model.predict(X)
        decoded_labels = GreedyDecoder(batch_logits)
        batch_predictions = get_batch_transcripts(decoded_labels)
        batch_transcripts = ["".join([idx_mapping[i] for i in label]) for label in y]
        
        predictions.extend(batch_predictions)
        targets.extend(batch_transcripts)
        wer_score = wer(batch_transcripts, batch_predictions)
        mean_wer += wer_score
        batch_num += 1
    mean_wer = mean_wer/batch_num
    print(f"\033[1m CTC LOSS: {round(loss, 2)} \033[0m")
    print(f"\033[1m WER: {int(wer_score*100)}% \033[0m")
    print("*"*70)
    for i in np.random.randint(0, len(predictions), 5):
        print(f"\033[1m Transcript: \033[0m {targets[i]}")
        print(f"\033[1m prediction: \033[0m {predictions[i]}")
        print("*"*70)
    return loss, mean_wer

def visualization_noise(noise_sound, df):
    noise_rates = [0.1, 0.5,0.9]
    i = np.random.randint(0, len(df))
    #Speech sound
    filename = df["filename"].iloc[i]
    sf, audio = wavfile.read(f"./data/LJSpeech-1.1/wavs/{filename}.wav")
  
    print(f"\033[1m  - Speech audio")
    display.display(display.Audio(np.transpose(audio), rate=16000))
    transcript = df["transcript"].iloc[i]
    print(f"\033[1m  - Transcript: {transcript}")
    plt.plot(audio)
    plt.title("Signal wave of speech audio")
    plt.xlim(0, len(audio))
    plt.show()
    
    #Trim the noise sound
    i = np.random.randint(int(len(noise_sound) - len(audio)))
    try:
      noise_trim = noise_sound[i:i+len(audio)][:, 0]
    except:
      noise_trim = noise_sound[i:i+len(audio)]

    print("\033[1m  - Noise audio")
    display.display(display.Audio(np.transpose(noise_trim), rate=16000))
    plt.plot(noise_trim)
    plt.title("Signal wave of Noise sound")
    plt.xlim(0, len(noise_trim))
    plt.show()
    
    #Add battle backround to speech
    for noise_rate in noise_rates:
        mix_audio =  audio*(1-noise_rate) + noise_trim*noise_rate
        print(f"\033[1m  - Mixed audio with noise_rate = {noise_rate}")
        plt.plot(mix_audio)
        plt.title("Signal Wave of mixed audio")
        plt.xlim(0, len(mix_audio))
        display.display(display.Audio(np.transpose(mix_audio), rate=16000))
        plt.show()
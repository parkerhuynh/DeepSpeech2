import pandas as pd
import numpy as np
import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import keras
import config
from scipy.io import wavfile
import python_speech_features


class CleanDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset, data_parameter = config.data_parameters,
                chr_mapping = utils.chr_mapping()):
        'Initialization'
        self.dataset = dataset
        self.batch_size = data_parameter["batch_size"]
        self.char_mapping = chr_mapping
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_data = [self.dataset.iloc[k] for k in indexes]
        audios, labels = self.__data_generation(batch_data)
        return audios, labels

        'text processing'
    def text_to_idx(self, text):
        text  = text.lower()
        idx = []
        for chr in text:
            if chr in self.char_mapping :
                idx.append(self.char_mapping[chr])
        return idx
          
        'normalize raw audio'
                
    def normalize(self, audio):
        gain = 1.0 / (np.max(np.abs(audio)) + 1e-5)
        return audio * gain
        
        'standardize FBANK'
    def standardize(self,features):
        mean = np.mean(features)
        std = np.std(features)
        return (features - mean) / std
        
        'FBAnk processing'
    def audio_to_features(self, audio):
        sf, audio = wavfile.read(f"./data/LJSpeech-1.1/wavs/{audio}.wav")
        audio = self.normalize(audio.astype(np.float32))
        audio = (audio * np.iinfo(np.int16).max).astype(np.int16)
        feat, energy = python_speech_features.fbank(
            audio, nfilt=160, winlen=0.02,winstep=0.01, winfunc = np.hanning)
        features = np.log(feat)
        return  self.standardize(features)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataset))

    def __data_generation(self, batch_data):
        audios = []
        labels = []
        label_len = []
        audio_len = []

        for filename, transcript in batch_data:
            audio = self.audio_to_features(filename)
            audios.append(audio)
            audio_len.append(len(audio))

            label = self.text_to_idx(transcript)
            labels.append(label)
            label_len.append(len(label))
            
        max_audio_len = max(audio_len)
        max_label_len = max(label_len)
        audios = pad_sequences(audios, maxlen = max_audio_len, dtype='float32', value=0, padding='post')
        labels = pad_sequences(labels, maxlen = max_label_len, value=28, padding='post')
        return audios, labels

class NoiseDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset, noise_sound, batch_size=32, noise_rate = config.data_parameters["noise_rate"]):
        'Initialization'
        self.dataset = dataset
        self.batch_size = data_parameter["batch_size"]
        self.char_mapping = chr_mapping
        self.noise_sound = noise_sound
        self.noise_rate = noise_rate
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_data = [self.dataset.iloc[k] for k in indexes]
        audios, labels = self.__data_generation(batch_data)
        return audios, labels

        'text processing'
    def text_to_idx(self, text):
        text  = text.lower()
        idx = []
        for chr in text:
            if chr in self.char_mapping :
                idx.append(self.char_mapping[chr])
        return idx
        'normalize raw audio'
                
    def normalize(self, audio):
        gain = 1.0 / (np.max(np.abs(audio)) + 1e-5)
        return audio * gain
        
        'standardize FBANK'
    def standardize(self,features):
        mean = np.mean(features)
        std = np.std(features)
        return (features - mean) / std
        
        'FBAnk processing'
    def audio_to_features(self, audio):
        #Load speech sound 
        sf, audio = wavfile.read(f"./LJSpeech-1.1/wavs/{audio}.wav")
        #Trim battle sound
        i = np.random.randint(int(len(self.noise_sound) - len(audio)))
        try:
          trim_audio = self.noise_sound[i:i+len(audio)][:, 0]
        except:
          trim_audio = self.noise_sound[i:i+len(audio)]
        #Add noise sound to speech sound
        audio =  audio*(1-self.noise_rate) + trim_audio*self.noise_rate
        audio = self.normalize(audio.astype(np.float32))
        audio = (audio * np.iinfo(np.int16).max).astype(np.int16)
        feat, energy = python_speech_features.fbank(
            audio, nfilt=160, winlen=0.02,winstep=0.01, winfunc = np.hanning)
        features = np.log(feat)
        return  self.standardize(features)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataset))

    def __data_generation(self, batch_data):
        audios = []
        labels = []
        label_len = []
        audio_len = []

        for filename, transcript in batch_data:
            audio = self.audio_to_features(filename)
            audios.append(audio)
            audio_len.append(len(audio))

            label = self.text_to_idx(transcript)
            labels.append(label)
            label_len.append(len(label))
            
        max_audio_len = max(audio_len)
        max_label_len = max(label_len)
        audios = pad_sequences(audios, maxlen = max_audio_len, dtype='float32', value=0, padding='post')
        labels = pad_sequences(labels, maxlen = max_label_len, value=28, padding='post')
        return audios, labels
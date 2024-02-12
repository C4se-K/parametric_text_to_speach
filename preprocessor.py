import os
import glob
import soundfile as sf
import wanakana
import librosa
import numpy as np
import MeCab

class preprocessor:
    def __init__(self, target_sr=22050, n_mels=256, hop_length=512, n_fft=2048):
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft

    def run(self):
        dataset = self.preprocess_audio('source')
        dataset = self.preprocess_to_mel(dataset)

        
        #for item in dataset[:5]:
            #print("Transcription:", item[1])

        dataset = self.tokenize_int(dataset)
        #print(len(dataset))
        dataset = self.remove_error(dataset)
        #print(len(dataset))
        #currently set to 5 for testing
        """
        for item in dataset: #[:5]:
            print("Transcription:", item[1])
            print("Audio Data Length:", len(item[0]), "samples")
            #print(wanakana.to_katakana(item[1]))
        """
        
        return dataset

    #returns audio bytes and transcriptions from file
    def preprocess_audio(self, file_name):
        #hardcoded data path
        path = os.path.join(os.getcwd(), 'data', file_name)
        dataset = []

        #recurively walk through directory
        for root, dirs, files in os.walk(path):
            for file in files[:5]:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    data, samplerate = sf.read(file_path)
                    
                    # format is "transcription_0x.wav"
                    transcription = file.rsplit('_', 1)[0]
                    dataset.append((data, wanakana.to_katakana(transcription)))
        
        return dataset

    #returns a tuple with mel frequency spec of audio and transcription
    def preprocess_to_mel(self, dataset, max_len=None):
        dataset_mel = []
        max_length = 0  # Track the maximum length for padding if needed

        for data, transcription in dataset:
            # Trim silence from the beginning and the end
            data, _ = librosa.effects.trim(data)
                
            # Convert to mel spectrogram
            spectrogram = librosa.feature.melspectrogram(y=data, sr=self.target_sr, n_mels=self.n_mels,
                                                        hop_length=self.hop_length, n_fft=self.n_fft)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

            # Update max_length if this spectrogram is longer
            if max_len is None and spectrogram_db.shape[1] > max_length:
                max_length = spectrogram_db.shape[1]

            dataset_mel.append((spectrogram_db, transcription))

        # If a maximum length is specified or determined, pad all spectrograms to this length
        if max_len is not None or max_length > 0:
            padded_dataset_mel = []
            target_length = max_len if max_len is not None else max_length
            for spectrogram_db, transcription in dataset_mel:
                # Calculate the required padding
                padding_amount = target_length - spectrogram_db.shape[1]
                if padding_amount > 0:
                    # Pad the spectrogram to the right with the minimum value of the spectrogram to keep dB scale consistent
                    padded_spectrogram = np.pad(spectrogram_db, ((0, 0), (0, padding_amount)), mode='constant', constant_values=(spectrogram_db.min(),))
                else:
                    padded_spectrogram = spectrogram_db
                padded_dataset_mel.append((padded_spectrogram, transcription))
            return padded_dataset_mel

        return dataset_mel


    def tokenize_int(self, dataset):
        

        dataset_tokens = []

        for mel, word in dataset:
            word_tokens = [self.get_kana_mapping(char) for char in word]
            #print(word)
            dataset_tokens.append((mel, word_tokens))

        return dataset_tokens
    
    def get_kana_mapping(self, kana):
        jp_map = {
            # Basic vowels
            'ア': 1, 'イ': 2, 'ウ': 3, 'エ': 4, 'オ': 5,
            # K series
            'カ': 6, 'キ': 7, 'ク': 8, 'ケ': 9, 'コ': 10,
            'ガ': 11, 'ギ': 12, 'グ': 13, 'ゲ': 14, 'ゴ': 15,
            # S series
            'サ': 16, 'シ': 17, 'ス': 18, 'セ': 19, 'ソ': 20,
            'ザ': 21, 'ジ': 22, 'ズ': 23, 'ゼ': 24, 'ゾ': 25,
            # T series
            'タ': 26, 'チ': 27, 'ツ': 28, 'テ': 29, 'ト': 30,
            'ダ': 31, 'ヂ': 32, 'ヅ': 33, 'デ': 34, 'ド': 35,
            # N series
            'ナ': 36, 'ニ': 37, 'ヌ': 38, 'ネ': 39, 'ノ': 40,
            # H series
            'ハ': 41, 'ヒ': 42, 'フ': 43, 'ヘ': 44, 'ホ': 45,
            'バ': 46, 'ビ': 47, 'ブ': 48, 'ベ': 49, 'ボ': 50,
            'パ': 51, 'ピ': 52, 'プ': 53, 'ペ': 54, 'ポ': 55,
            # M series
            'マ': 56, 'ミ': 57, 'ム': 58, 'メ': 59, 'モ': 60,
            # Y series
            'ヤ': 61, 'ユ': 62, 'ヨ': 63,
            # R series
            'ラ': 64, 'リ': 65, 'ル': 66, 'レ': 67, 'ロ': 68,
            # W series and N
            'ワ': 69, 'ヰ': 70, 'ヱ': 71, 'ヲ': 72, 'ン': 73,
            # Small versions of vowels, ya, yu, yo, and tsu
            'ァ': 74, 'ィ': 75, 'ゥ': 76, 'ェ': 77, 'ォ': 78,
            'ャ': 79, 'ュ': 80, 'ョ': 81,
            # Special marks
            'ッ': 82,
            'ー': 83,  # Long vowel sound
        }
        
        return jp_map.get(kana, 0)
    
    def remove_error(self, dataset):
        dataset_temp = []
        for mel, word in dataset:
            zero_found = False
            for char in word:
                if char == 0:
                    zero_found = True
                    break
            if not zero_found:
             dataset_temp.append((mel, word))
        return dataset_temp
    
    """
    
    jp_map = {
            # Basic vowels
            'ア': 1, 'イ': 2, 'ウ': 3, 'エ': 5, 'オ': 7,
            # K series
            'カ': 11, 'キ': 22, 'ク': 33, 'ケ': 55, 'コ': 77,
            'ガ': 13, 'ギ': 26, 'グ': 39, 'ゲ': 65, 'ゴ': 91,
            # S series
            'サ': 17, 'シ': 34, 'ス': 51, 'セ': 85, 'ソ': 119,
            'ザ': 19, 'ジ': 38, 'ズ': 57, 'ゼ': 95, 'ゾ': 133,
            # T series
            'タ': 23, 'チ': 46, 'ツ': 69, 'テ': 115, 'ト': 161,
            'ダ': 29, 'ヂ': 58, 'ヅ': 87, 'デ': 145, 'ド': 203,
            # N series
            'ナ': 31, 'ニ': 62, 'ヌ': 93, 'ネ': 155, 'ノ': 217,
            # H series
            'ハ': 37, 'ヒ': 74, 'フ': 111, 'ヘ': 185, 'ホ': 259,
            'バ': 41, 'ビ': 82, 'ブ': 123, 'ベ': 205, 'ボ': 287,
            'パ': 43, 'ピ': 86, 'プ': 129, 'ペ': 215, 'ポ': 301,
            # M series
            'マ': 47, 'ミ': 94, 'ム': 141, 'メ': 235, 'モ': 329,
            # Y series
            'ヤ': 53, 'ユ': 159, 'ヨ': 371,
            # R series
            'ラ': 59, 'リ': 118, 'ル': 295, 'レ': 331, 'ロ': 413,
            # W series and N
            'ワ': 61, 'ヰ': 122, 'ヱ': 305, 'ヲ': 359, 'ン': 427,
            # Small versions of vowels, ya, yu, yo, and tsu
            'ァ': 67, 'ィ': 134, 'ゥ': 335, 'ェ': 389, 'ォ': 469,
            'ャ': 71, 'ュ': 213, 'ョ': 355,
            # Special marks
            'ッ': 73,
            'ー': 79,  # Long vowel sound
        }
    """
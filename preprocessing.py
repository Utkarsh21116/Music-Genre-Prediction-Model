import librosa
import numpy as np

def preprocessing(file_path,SAMPLE_RATE=22050,DURATION=3,N_MELS=128,DESIRED_LEN=128):

    samples_per_segment = SAMPLE_RATE * DURATION

    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

    sample_per_track = signal.shape[0]
    num_segments = max(1, sample_per_track // samples_per_segment)
    mel_segments = []

    for s in range(num_segments):

        start = s * samples_per_segment
        end = start + samples_per_segment

        segment = signal[start:end]

        # pad short segments
        if len(segment) < samples_per_segment:
            segment = librosa.util.fix_length(
                segment,
                size=samples_per_segment
            )

        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=SAMPLE_RATE,
            n_mels=N_MELS
        )

        # Convert to dB
        #mel = librosa.power_to_db(mel, ref=np.max)

        # transpose -> (time, frequency)
        mel = mel.T

        # fix shape
        if mel.shape[0] < DESIRED_LEN:
            pad_width = DESIRED_LEN - mel.shape[0]
            mel = np.pad(
                mel,
                ((0, pad_width), (0, 0)),
                mode='constant')
        else:
            mel = mel[:DESIRED_LEN, :]

        mel_segments.append(mel)

    mel_segments = np.array(mel_segments)

    # add channel dimension
    mel_segments = mel_segments[..., np.newaxis]

    return mel_segments

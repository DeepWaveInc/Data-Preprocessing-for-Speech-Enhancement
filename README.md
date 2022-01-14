# 📝 The Procedure of Data Preprocessing for Speech Enhancement (Self-attention U-Net6)

## 🧩 Resource

- Noise datasets
  - Youtube (Audioset. [repo for processing](https://github.com/WangWilly/audioset-processing)): Extract audio datas from the given list `balanced_train_segments.csv`, classes including "Noise", "Environmental noise", "White noise", "Dog", "Cat" and "Musical instrument".
  - musan-noise (https://www.openslr.org/17/): Utilize two given subsets: free-sound and sound-bible.
- Speech dataset
  - TCC300

## 🛠 Python Code

We use three pieces of Python code to achieve the training data.
- One is for spliting resource data into subsets including `train`, `validate` and `test`, and ratios among them are *0.7*, *0.15*, *0.15*.
- Another is for concatenating all noise clips to `train.wav` and `test.wav`.
- The last is for generating chunks in `.h5` format. In order to make training procedure painless, designing a proper duration of the input audio and adjusuting the noise SNR are achieved here.

### 📑 Split data

The designed function `split_data` splits the original dataset into three subsets `train`, `validate` and `test` and requires three augments:
- `dir`: By utilizing `librosa.util.find_files(...)`, all audio files contained in the given directory will be search out. Then, all files are shuffled randomly. (`np.random.seed(87)`)
- `name`: [TODO -> deprecate?] All original file paths respected to `train`, `validate` and `test` are kept in the numpy file `filepath_{name}.npy`. The file is named after this augment.
- `type`: [TODO -> deprecate?] `filepath_{name}.npy` will be saved at this folder. (=> `tpye/filepath_{name}.npy`)
- Although `output_dir` is not a augment of `split_data()`, it is still the necessery to provide. 

```python
import librosa
import os
import numpy as np
from pathlib import Path
from shutil import copyfile

np.random.seed(87)


def split_data(dir, name, type):
    files = librosa.util.find_files(dir)
    output_dir = '/path/to/store/splited-data/'
    os.makedirs(output_dir, exist_ok=True)
    np.random.shuffle(files)

    os.makedirs(f'./{type}', exist_ok=True)

    train = []
    validate = []
    test = []

    train_num = int(len(files) * 0.7)
    validate_num = int(len(files) * 0.15)

    for idx, f in enumerate(files):
        if idx <= train_num:
            train.append(f)
        elif train_num < idx <= train_num + validate_num:
            validate.append(f)
        else:
            test.append(f)

    data = {
        'train': train, 
        'validate': validate, 
        'test': test
    }
    np.save(os.path.join(type, f'filepath_{name}.npy'), data)

    os.makedirs(os.path.join(output_dir, 'train'), exists_ok=True)
    for f in train:
        copyfile(f, os.path.join(output_dir, 'train', f'{Path(f).name}'))

    os.makedirs(os.path.join(output_dir, 'validate'), exists_ok=True)
    for f in validate:
        copyfile(f, os.path.join(output_dir, 'validate', f'{Path(f).name}'))

    os.makedirs(os.path.join(output_dir, 'test'), exists_ok=True)
    for f in test:
        copyfile(f, os.path.join(output_dir, 'test', f'{Path(f).name}'))


if __name__ == '__main__':
    split_data('/path/to/tcc300/WAV/', 'tcc300', 'speech')
    # OR
    split_data('/path/to/musan/noise/free-sound', 'muasn_free-sound', 'noise')
    # OR
    split_data('/path/to/musan/noise/sound-bible', 'musan_sound-bible', 'noise')

    split_data(
        '____/path/to/dataset/____',
        '__[TODO: deprecate] name of the path list__',
        '__[TODO: deprecate] folder name for saving path list file__'
    )
```

### 📑 Rearrange noise data for training and testing

This piece of Python collects all data from subsets. In here, `train.wav` is made up by concatenating audio data coming from the `train` and the `validate` subset. On the other hand, `test.wav` is made up by concatenating audio data coming from the `test` subset.

```python
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


noise_train_data_dirs = [
    '/path/to/audioset_splited-train',
    '/path/to/muasn_free-sound_noise_splited-train',
    '/path/to/musan_sound-bible_noise_splited-train'
]
noise_validate_data_dirs = [
    '/path/to/audioset_splited-validate',
    '/path/to/muasn_free-sound_noise_splited-validate',
    '/path/to/musan_sound-bible_noise_splited-validate'
]
noise_test_data_dirs = [
    '/path/to/audioset_splited-test',
    '/path/to/muasn_free-sound_noise_splited-test',
    '/path/to/musan_sound-bible_noise_splited-test'
]

DIRS = noise_train_data_dirs + noise_validate_data_dirs 
# OR
DIRS = noise_test_data_dirs

output_name = 'train.wav'
# OR
output_name = 'test.wav'

SAMPLE_RATE = 16000
output_dir = Path('/path/to/store/noise-data')


if __name__ == '__main__':
    noise_files = []
    for d in DIRS:
        noise_files.extend(librosa.util.find_files(d))

    audio_data = np.ndarray(shape=(0,), dtype='float32')
    for f in tqdm(noise_files, ascii=True):
        y, sr = librosa.load(f, sr=SAMPLE_RATE, mono=True)
        audio_data = np.concatenate([audio_data, y], axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(output_dir / output_name, audio_data, SAMPLE_RATE)
```

### 📑 Generate noisy data (in both h5 and wav format)

This piece of Python allows multi-thread generating chunks for training, and it is possible to tweak the number of thread for processing. In here, the paths to `train.wav`, `test.wav` and the TCC300 directory (which is already splited into `train`, `validate` and `test`) are required. Furthermore, depending on the DL structure, the following augments are availible to change.

In order to generate the input of Self-attention U-Net6, its input shape is `(512, 256)`, applying following augments:
- `CHUNK_DURATION`: 5 seconds.
- The scalars of noise SNR: `[0, -10]`.

🩲 Algorithm in brief:
```text
- Decide the thread pool.
- Provide noise data: `train.wav`, `test.wav`.
- Provide clean speech: TCC300.
- For each speech audio:
  1. Randomly extract the noise from `train.wav` (or `test.wav` when generating test data) and the length of noise is the same as the speech audio.
  2. Decide the noise SNR (Only 0 and -10 are the option in our prior case) and mixture them to a noisy speech.
  3. Separate both noisy speechs and original speechs into 5-secs chunks. If the last remaining is greater than 2 secs, it is still kept as a chunk.
```

In this case, the processed files are still in waveform. When the spectrgram is required in the training procedure, the STFT (`librosa.stft`) augments such as `n_fft=1024` and `hop_length=256` are still needed to provide.

After the process done, the files would be:
```text
noisy_speech_dataset/
├── test/
|   ├── h5/ => each files contain `5 * SAMPLE_RATE` samples in this case.
|   |   ├── M010101_0-snr_-10-0/ <─┐ This is the first five-seconds
|   |   |   ├── noisy_speech.h5    |  chunk of `M010101_0-snr_-10.wav`.
|   |   |   └── speech.h5          |
|   |   ├── ...                    |
|   |   └── M010101_0-snr_0-7/     |
|   └── wav/                       |
|       ├── M010101_0-snr_-10.wav ─┘
|       ├── ...
|       └── M010101_0-snr_0.wav => noisy speech in .wav format.
├── train/
└── validate/
```


```python
import h5py
import librosa
import numpy as np
import soundfile as sf

from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

np.random.seed(87)

CHUNK_DURATION = 5
SAMPLE_RATE = 16000

train_noise, _ = librosa.load("/path/to/Noise/train.wav", SAMPLE_RATE)
test_noise, _ = librosa.load("/path/to/Noise/test.wav", SAMPLE_RATE)

speech_dir = Path("/path/to/TCC300/tcc300_splited")
output_dir = Path("/path/to/noisy_tcc300")


def add_noise(signal, noise, snr):
    signal_length = len(signal) 
    noise_length = len(noise)
    pw_sig = np.sum(np.power(signal, 2)) / signal_length
    pw_noise = np.sum(np.power(noise, 2)) / noise_length
    scale = 10 * np.log10(pw_sig / pw_noise) - snr
    scale /= 10
    scale = np.sqrt(10**scale)
    result = signal + noise * scale
    return result


def process_one(filename, category, output_dir: Path):
    raw_wav, _ = librosa.load(filename, SAMPLE_RATE)
    if librosa.get_duration(raw_wav, SAMPLE_RATE) < CHUNK_DURATION:
        return

    if category == "test":
        noise = test_noise
    else:
        noise = train_noise
    
    noise_idx = np.random.randint(0, noise.shape[0] - raw_wav.shape[0])
    wav_dir = output_dir / "wav"
    h5_dir = output_dir / "h5"
    wav_dir.mkdir(parents=True, exist_ok=True)
    h5_dir.mkdir(parents=True, exist_ok=True)

    for snr in [0, -10]:
        processed_wav = add_noise(raw_wav, noise[noise_idx:noise_idx + raw_wav.shape[0]], snr)
        sf.write(wav_dir / f"{Path(filename).stem}-snr_{snr}.wav", processed_wav, SAMPLE_RATE)
        
        nb_chunk = 0
        current_idx = 0
        while current_idx + CHUNK_DURATION * SAMPLE_RATE < processed_wav.shape[0]:
            h5_target_dir = h5_dir / f'{Path(filename).stem}-snr_{snr}-{nb_chunk}'
            h5_target_dir.mkdir(parents=True, exist_ok=True)

            with h5py.File(h5_target_dir / "speech.h5", "w") as f:
                f.create_dataset("dataset", data=raw_wav[current_idx:current_idx + CHUNK_DURATION * SAMPLE_RATE])
                f.create_dataset("dataset_len", data=np.array([CHUNK_DURATION * SAMPLE_RATE]))
                f.create_dataset("sample_rate", data=np.array([SAMPLE_RATE]))

            with h5py.File(h5_target_dir / "noisy_speech.h5", "w") as f:
                f.create_dataset("dataset", data=processed_wav[current_idx:current_idx + CHUNK_DURATION * SAMPLE_RATE])
                f.create_dataset("dataset_len", data=np.array([CHUNK_DURATION * SAMPLE_RATE]))
                f.create_dataset("sample_rate", data=np.array([SAMPLE_RATE]))

            current_idx += CHUNK_DURATION * SAMPLE_RATE
            nb_chunk += 1
            
        if processed_wav.shape[0] - current_idx > int(2 * SAMPLE_RATE):
            h5_target_dir = h5_dir / f'{Path(filename).stem}-snr_{snr}-{nb_chunk}'
            h5_target_dir.mkdir(parents=True, exist_ok=True)

            with h5py.File(h5_target_dir / "speech.h5", "w") as f:
                f.create_dataset("dataset", data=raw_wav[raw_wav.shape[0] - CHUNK_DURATION * SAMPLE_RATE:])
                f.create_dataset("dataset_len", data=np.array([CHUNK_DURATION * SAMPLE_RATE]))
                f.create_dataset("sample_rate", data=np.array([SAMPLE_RATE]))

            with h5py.File(h5_target_dir / "noisy_speech.h5", "w") as f:
                f.create_dataset("dataset", data=processed_wav[raw_wav.shape[0] - CHUNK_DURATION * SAMPLE_RATE:])
                f.create_dataset("dataset_len", data=np.array([CHUNK_DURATION * SAMPLE_RATE]))
                f.create_dataset("sample_rate", data=np.array([SAMPLE_RATE]))


if __name__ == "__main__":
    for category in ["train", "validate", "test"]:
        p = Pool(40)

        files = librosa.util.find_files(speech_dir / category)
        pbar = tqdm(total=len(files), bar_format='{desc}{percentage:3.0f}%|{bar:5}{r_bar}')

        def update(*a):
            pbar.update()

        target_dir = output_dir / category
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            if category == "test":
                p.apply_async(process_one, args=(f, category, target_dir), callback=update)
            else:
                p.apply_async(process_one, args=(f, category, target_dir), callback=update)

        p.close()
        p.join()
```

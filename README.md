# VoiceBiometrics
Speaker recognition with MFCC and neural networks using VoxCeleb dataset. Below is the overwiev of the modules and their major classes.

### vb.py
Main module for running the training with `run_experiment`. For every HPS run `Trainer` creates and trains a `Model` for each permutation in k-fold validation.

### data.py
For every k-fold permutation the `DataBuilder` creates train-validation datasets pair and `InputNormalizer`. `InputNormalizer` transforms every MFCC channel to standard normal distribution based on training dataset. MFCC tensors for every file, metadata and `InputNormalizer` parameters are cached on disk to avoid recomputing. Since the audio files in VoxCeleb are produced by cutting multiple larger recordings, in order to avoid leaks, the files are split based on the recording of origin.

### models.py
Includes architectures to build networks based on the config.

### monitors.py
For every HPS run an `ExperimentMonitor` combines and processes training metrics collected for every fold by a `ScoresMonitor`.

### hps.py
For every HPS run `HPSParser` generates a config by drawing a random element from every `HPSVariant` contained in the base config.

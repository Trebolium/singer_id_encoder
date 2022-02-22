<h1>Implementation of Speaker Verification model for Singing</h1>

<h2>Brendan O'Connor, Simon Dixon, George Fazekas</h2>

<h2>Centre For Digital Music, Queen Mary University of London, UK</h2>

This repo consists of a network designed to capture voice identity embeddings which uses an LSTM stack and the generalised end-to-end loss which enforces similarity between all embeddings for the same speaker. 

In our implementation, we have added multiple functionalities which include:

1. An iteration protocol that cycles between training and validation data until validation no longer improves or maximum iterations have been completed
2. The ability to use either or both GE2E and classification loss
2. Recording these metrics for analysis in tensorboard
3. Generating a directory that contains both the configuration parameters and model checkpoint data
4. Generating a directory within a parent directory to reflect the history of training when using pretrained weights from a previous models `saved_model.pt` checkpoint.

These functionalities have allowed us to explore feature-engineering, measuring the effects of different inputs on the performance of this network in the context of the singing domain. Reports of this will be presented in our upcoming paper.


To train the network at default settings, simply run:
```
python main.py
```

Please note the default settings utilise the _example_dataset_ which is a minimal example of VocalSet data converted into melspectrogram features. To change this, use the `-fd` flag to specificy a new directory. Please note that the directory must be split into subsets labelled 'train' and 'val', and must contain a feat_params.yaml similar to the one in the supplied 'example_dataset' directory.

```
python main.py -fd=path/to/dataset
```

<h2>Acknowledgments</h2>

In our implementation, we have used the model implemented by [CorentinJ](https://github.com/CorentinJ/Real-Time-Voice-Cloning) which reflects the architecture proposed by [Wan et al. 2018](10.1109/ICASSP.2018.8462665).
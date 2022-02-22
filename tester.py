from re import I
from data_objects.random_cycler import RandomCycler
from data_objects.utterance import Utterance
import numpy as np
import pdb

def collater(dataset, utterances_per_speaker, n_frames, num_feats):
    # warning: If using RandomCycler, dataset will be infinite
    for i, (speaker, idx) in enumerate(dataset):
        try:
            spkr_uttrs_list = [str(f)[:-4] for f in speaker.root.glob('*.npy')]
            utterances = [Utterance(speaker.root.joinpath(f+'.npy'), f+'.wav') for f in spkr_uttrs_list]
            print(f"Number of utterances for speaker {speaker.name} is: {len(utterances)}")
            utterance_cycler = RandomCycler(utterances)
        except:
            pdb.set_trace()
    speakers_data = [dataset[2], dataset[4], dataset[7]]
    partials = {s.name: s.random_partial(utterances_per_speaker, n_frames) for s,_ in speakers_data}
    x_data = np.array([uttr_data[1] for s,_ in speakers_data for uttr_data in partials[s.name]])
    y_data = np.array([speakers_data[i//utterances_per_speaker][1] for i in range(len(x_data))])
    
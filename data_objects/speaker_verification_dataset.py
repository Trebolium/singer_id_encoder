import pdb
from data_objects.random_cycler import RandomCycler
from data_objects.speaker_batch import SpeakerBatch
from data_objects.speaker import Speaker
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# collects paths to utterances of speakers - does not collect the data itself
class SpeakerVerificationDataset(Dataset):
    def __init__(self, datasets_root: Path, partials_n_frames):
        self.partials_n_frames = partials_n_frames
        self.root = datasets_root
        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir() and not str(f).startswith('.')]
        if len(speaker_dirs) == 0:
            raise Exception("No speakers found. Make sure you are pointing to the directory "
                            "containing all preprocessed speaker directories.")
        self.speakers = [(Speaker(speaker_dir), i) for i, speaker_dir in enumerate(speaker_dirs)]
        self.num_speakers = len(self.speakers)
        self.speaker_cycler = RandomCycler(self.speakers)

    def num_voices(self):
        return self.num_speakers

    def __len__(self):
        return int(1e10) # so that when iterating over the loader it has a (close to) infinite amount of steps to do 
        
    def __getitem__(self, index):
        """ speaker_cycler chooses a random speaker from dataset (seemingly ignoring the index variable)
        The speaker_cycler assures that this randomness has some logical restrainsts
        """
        return next(self.speaker_cycler)
    
    def get_logs(self):
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += "".join(log_file.readlines())
        return log_string
    
    
class SpeakerVerificationDataLoader(DataLoader):
    def __init__(self, dataset, speakers_per_batch, utterances_per_speaker, partials_n_frames,
                    num_feats, sampler=None, batch_sampler=None, num_workers=0,
                    pin_memory=False, timeout=0, worker_init_fn=None):
        self.partials_n_frames = partials_n_frames
        self.utterances_per_speaker = utterances_per_speaker
        self.num_feats = num_feats
        
        super().__init__(
            dataset=dataset, 
            batch_size=speakers_per_batch, 
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers,
            collate_fn=self.collate, # collate converts everything into a tensor where the first dimension is the batch dimension
            pin_memory=pin_memory, 
            drop_last=False, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn
        )

    def collate(self, speaker_data):
        """This function used only when batch is called from dataloader. SpeakerBatch gets data from paths"""
        return SpeakerBatch(speaker_data, self.utterances_per_speaker, self.partials_n_frames, self.num_feats) 
    
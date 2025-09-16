import numpy as np
from matplotlib import pyplot as plt
import librosa # instalar com pip
import sounddevice as sd # instalar com pip

class Track:
    def __init__(self,label,filename=None):
        self.label = label
        self.filename = label+'.wav' if filename is None else filename
        self.data = None
        self.rate = 0
        self.n_steps = 0
        self.duration = 0
        self.load()
        
    def load(self):
        y, sr = librosa.load(self.filename)
        self.data = y if len(y.shape)==1 else y[0] # evita erros com stereo
        self.rate = sr
        self.n_steps = len(self.data)
        self.duration = self.n_steps / self.rate
        

track_labels = [
  'drums',
  'bass',
  'rhythm',
  'lead'
]

tracks = [ Track(label) for label in track_labels ]

sample_rate = tracks[0].rate # assumindo que todas as tracks tem o mesmo sample rate

A = np.array( [ track.data for track in tracks ] ).T

print(f'Tracks: {track_labels}')
print(f'Provide space separated weights for each of the {len(track_labels)} tracks')
while True:  
  ans = input("Next mix (q to quit)? ")
  if ans.lower() == 'q':
    exit()
  mix_str = ans.split(' ')
  if len(mix_str) != len(tracks):
    print("Invalid.")
    continue
  mix = np.array( [float(w) for w in mix_str] )
  song = A @ mix # a música é o resultado de um produto matriz-vetor! Ou seja, está no espaço coluna de A.
  sd.play( song, sample_rate )
  #sd.wait()
  _ = input("Press any key to stop and continue")
  sd.stop()


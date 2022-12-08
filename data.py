import torch 
import os
import muspy
import music21
import numpy as np

#The muspy implementation of .to_pytorch_dataset() returns a class which is not pickleable,
# so we define our own class which is pickleable.

### Define a dataset class in torch.utils.data.Dataset format.
### Data is stored as .midi files in a folder structure.
### The dataset class will need to convert the .midi files to pianoroll format using muspy.
  
class MidiTokenDataset(torch.utils.data.Dataset):
    def __init__(self, files, indices):
        self.files = files
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        #use music21 to tokenize the midi file
        print("Beginning parse")
        try:
            midi = music21.converter.parse(self.files[self.indices[index]])
            notes = midi.flat.getElementsByClass('Note')
        except:
            return None
        
        #128 pitches
        pitches = np.zeros((len(notes)), dtype=np.float32)
        
        #durations are quantized into bins with 1/32nd note resolution
        #A quarter note has duration 1 in midi, under our quantization scheme, a quarter note has duration 8
        durations = np.zeros((len(notes)), dtype=np.int32)
        
        #offset from bar start, quantized to 1/32nd note resolution
        positions = np.zeros((len(notes)), dtype=np.int32) 
        
        #a bar is no longer than two whole notes: so 64 1/32nd notes
        bars = np.zeros((len(notes)), dtype=np.int32)
        timeNumerator = np.zeros((len(notes)), dtype=np.int32)
        timeDenominator = np.zeros((len(notes)), dtype=np.int32)
        instruments = np.zeros((len(notes)), dtype=np.int32)
        tempo = np.zeros((len(notes)), dtype=np.float32) #tempo almost certainly less than 320. Discretize into 32 bins.
        
        #velocity is between 0 and 127, we break that range up into 32 bins
        velocity = np.zeros((len(notes)), dtype=np.float32)
        
        #0: no special token, 1: mask, 2: pad
        specialtoken = np.zeros((len(notes)), dtype=np.int32)
        
        cumulativeOffset = 0
        #4/4
        lastBar = 0
        ts = notes[0].getContextByClass('TimeSignature')
        lastTimeSignature = ts.numerator / ts.denominator
        
        print("Beginning tokenization")
        for i, note in enumerate(notes):
            print("tokenizing note " + str(i) +" of " + str(len(notes)))
            pitches[i] = note.pitch.midi #at most 128
            if note.duration is not None:
                durations[i] = int(note.duration.quarterLength * 8 or 8) #at most 128
            else:
                durations[i] = 8 #default to quarter note
            sig = note.getContextByClass('TimeSignature')
            if sig is not None:
                timeNumerator[i] = sig.numerator or 4
                timeDenominator[i] = sig.denominator or 4
            else:
                timeNumerator[i] = 4
                timeDenominator[i] = 4 #default to 4/4
                
            bars[i] = int(note.measureNumber or 0) #support up to 1024 bars
            if bars[i] != lastBar:
                barDiff = bars[i] - lastBar
                cumulativeOffset += barDiff * lastTimeSignature * 4
                lastBar = bars[i]
                lastTimeSignature = timeNumerator[i] / timeDenominator[i]
                
            positions[i] = int(((note.offset or 0)  - cumulativeOffset)*8) #up to 64 1/32nd notes in a bar
                
            inst = note.getContextByClass('Instrument')
            instruments[i] = int(0)
            if inst is not None:
                instruments[i] = int(inst.midiProgram or 0) #up to 128 instruments
            met = note.getContextByClass('MetronomeMark')
            if met is not None:
                tempo[i] = int(met.number // 10 or 16)
            else:
                tempo[i] = 16 #default to 160 bpm
            velocity[i] = int(note.volume.velocity // 4 or 16) #127 / 4 = 32 bins
        
        
        
        pitches = torch.from_numpy(pitches).long() #1
        durations = torch.from_numpy(durations).long() #2
        positions = torch.from_numpy(positions).long() #3
        timeDenominator = torch.from_numpy(timeDenominator).long() #4
        timeNumerator = torch.from_numpy(timeNumerator).long() #5
        bars = torch.from_numpy(bars).long() #6
        instruments = torch.from_numpy(instruments).long() #7
        tempo = torch.from_numpy(tempo).long() #8
        velocity = torch.from_numpy(velocity).long() #9
        
        if pitches.shape[0] == 0:
            return None
        
        minbar = min(bars)
        bars = bars - minbar
        
        #clamp the values to be within the range of the vocabulary
        pitches = torch.clamp(pitches, 0, 127)
        durations = torch.clamp(durations, 0, 127)
        positions = torch.clamp(positions, 0, 31)
        timeDenominator = torch.clamp(timeDenominator, 0, 31)
        timeNumerator = torch.clamp(timeNumerator, 0, 31)
        bars = torch.clamp(bars, 0, 1023)
        instruments = torch.clamp(instruments, 0, 127)
        tempo = torch.clamp(tempo, 0, 31)
        velocity = torch.clamp(velocity, 0, 31)
        
        #ensure deletion to avoid memory leak
        del midi
        del notes
        
        return (pitches, durations, positions, timeDenominator, timeNumerator, bars, instruments, tempo, velocity, specialtoken)

class BERTTokenBatcher():
    def __call__(self, x):
        #x should be a list of tuples of tensors
        #if x is just a tuple, convert it to a list of length 1
 
        if type(x) == tuple:
            x = [x]

        x_stacked = [None, None, None, None, None, None, None, None, None, None]
        
        for z in x:
            if z is None:
                continue
            for i in range(len(z)):
                #pad the tensor to be a multiple of max_length
                if i == 9: #special token
                    #randomly mask 15% of the tokens
                    masked = (torch.rand(z[i].shape[0]) < 0.15).long()
                    padded = torch.nn.functional.pad(masked, (0, self.max_length - masked.shape[0] % self.max_length), value=2)
                else:
                    padded = torch.nn.functional.pad(z[i], (0, self.max_length - z[i].shape[0] % self.max_length), value=0)
                stacked = padded.reshape((-1, self.max_length))
                if x_stacked[i] is None:
                    x_stacked[i] = stacked
                else:
                    x_stacked[i] = torch.cat((x_stacked[i], stacked), dim=0)
        
        if x_stacked[0] is None:
            return None
        #take the minimum of the positions in each batch
        mins, _ = torch.min(x_stacked[5], dim=1, keepdim=True)
        x_stacked[5] = x_stacked[5] - mins
        
        return tuple(x_stacked)
       
        
    def __init__(self, max_length=256):
        self.max_length = max_length     
        
def midi(x):
    print(x)
    #x is a tuple of ten tensors: pitches, durations, positions, timeDenominator, timeNumerator, bars, instruments, tempo, velocity, specialtoken
    pitches, durations, positions, timeDenominator, timeNumerator, bars, instruments, tempo, velocity, specialtoken = x
    #if we have more than one example in the batch, just take the first one
    if len(pitches.shape) > 2:
        pitches = pitches[0]
        durations = durations[0]
        positions = positions[0]
        timeDenominator = timeDenominator[0]
        timeNumerator = timeNumerator[0]
        bars = bars[0]
        instruments = instruments[0]
        tempo = tempo[0]
        velocity = velocity[0]
        specialtoken = specialtoken[0]
    
    #then use music21 to convert to midi
    str_rep = ""
    
    s = music21.stream.Stream()
    s.timeSignature = music21.meter.TimeSignature(str(timeNumerator[0]) + "/" + str(timeDenominator[0]))
    lastBar = 0
    lastTimesig = s.timeSignature
    cumOffset = 0
    bar = music21.stream.Measure()
    bar.timeSignature = lastTimesig
    lastTempo = music21.tempo.MetronomeMark(number=tempo[0] * 10)
    str_rep += "<Tempo: " + str(lastTempo.number) + "> "
    str_rep += str(lastTimesig.numerator) + "/" + str(lastTimesig.denominator) + " "
    s.append(lastTempo)
    #sort the tokens by bar, then position
    pitches, durations, positions, timeDenominator, timeNumerator, bars, instruments, tempo, velocity, specialtoken = zip(*sorted(zip(pitches, durations, positions, timeDenominator, timeNumerator, bars, instruments, tempo, velocity, specialtoken), key=lambda x: (x[5], x[2])))
    
    for i in range(pitches.shape[0]):
        print("Processing token...")
        
        if specialtoken[i].item() == 1:
            str_rep += "<MASK> "
            continue
        if specialtoken[i].item() == 2:
            str_rep += "<PAD> "
            continue
        n = music21.note.Note(pitches[i].item())
        n.quarterLength = durations[i].item() / 8
        if bars[i].item() != lastBar:
            barDiff = bars[i].item() - lastBar
            lastBar = bars[i].item()
            
            for j in range(barDiff - 1):
                s.append(bar)
                str_rep += "| "
                bar = music21.stream.Measure()
                bar.timeSignature = lastTimesig
                cumOffset += bar.barDuration.quarterLength
                bar.offset = cumOffset
                bar.tempo = tempo[i].item() * 10
                
            lastTimesig = music21.meter.TimeSignature(str(timeNumerator[i].item()) + "/" + str(timeDenominator[i].item()))

        if lastTempo.number != tempo[i].item() * 10:
            lastTempo = music21.tempo.MetronomeMark(number=tempo[i].item() * 10)
            bar.append(lastTempo)
            str_rep += "<tempo: " + str(tempo[i].item() * 10) + "> "
        
        n.offset = positions[i].item() / 8
        n.volume.velocity = velocity[i].item() * 10
        n.storedInstrument = music21.instrument.fromString(instruments[i].item())
        bar.append(n)
        str_rep += str(n.pitch.nameWithOctave) + " "
        
    s.append(bar)
    print(str_rep)
    return s
    

def midi_dataset(split=0.95, lakh=False):
    paths = []
    root = "./data/maestro-v3.0.0"
    if lakh:
        root = "./data/lmd_full"
    for rt, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(".midi") or file.endswith(".mid"):
                #append full path
                paths.append(os.path.join(rt, file))

    trainidx = np.random.choice(len(paths), int(len(paths) * split), replace=False)
    testidx = np.setdiff1d(np.arange(len(paths)), trainidx)
    
    train = MidiTokenDataset(paths, trainidx)
    test = MidiTokenDataset(paths, testidx)
    
    return train, test

def download_lakh():
    #Load MAESTRO dataset into folder structure
    #This file is in: /home/username/....
    #We want: /scratch/username/....
    
    
    datapath = "/home/hliuson/data"
    _ = muspy.datasets.LakhMIDIDataset(root=datapath, download_and_extract=True)
    return getdatasets()
import numpy as np
import pretty_midi
import torch


def melody_sequence_to_index(seq, mapping):
    idx_seq = [mapping[i] for i in seq]
    idx_seq = torch.tensor([idx_seq])
    return idx_seq
    
def harmony_idx_to_sequence(idx_seq, mapping):
    harm_seq = np.array([np.array(mapping[i]) for i in idx_seq])
    return harm_seq

def reassemble_sequences(mel, harm, mel_dict, harm_dict):
    h = np.array([np.array(harm_dict[i]) for i in harm])
    m = np.array([mel_dict[i.item()] for i in mel.squeeze_()])
    #print("Function - melody: ", m)
    #print("Function - harmony: ", h)
    chorale = np.hstack((np.expand_dims(m, axis=1), h))
    return chorale

def piano_roll_from_sequence(seq, fs, cut_first=False):
    """
    Function to reinterpret MIDI integer encoding into pianoroll array
    """
    if cut_first:
        seq = seq[1:, :]
    seq_len = np.shape(seq)[0]
    piano_roll = np.zeros((128, seq_len), dtype=np.int32)
    for i in range(seq_len):
        piano_roll[seq[i, 0], i] = 96 #Set melody to higher velocity for better sound
        piano_roll[seq[i, 1], i] = 64
        piano_roll[seq[i, 2], i] = 64
        piano_roll[seq[i, 3], i] = 64
    return piano_roll
    
    
    
    
def piano_roll_to_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.

    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.

    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.

    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm
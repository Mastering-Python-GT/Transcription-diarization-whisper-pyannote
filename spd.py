from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper
import numpy as np
import gc

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization", use_auth_token="register to huggingface website to get the access token")


def read(k):
    y = np.array(k.get_array_of_samples())
    return np.float32(y) / 32768


def millisec(timeStr):
    spl = timeStr.split(":")
    return (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2])) * 1000)


k = str(pipeline(
    "Desktop/projects_python/tran_diar/audio.mp3")).split('\n')

del pipeline
gc.collect()

audio = AudioSegment.from_mp3(
    "Desktop/projects_python/tran_diar/audio.mp3")
audio = audio.set_frame_rate(16000)

model = whisper.load_model("small.en")

for l in range(len(k)):

    j = k[l].split(" ")
    start = int(millisec(j[1]))
    end = int(millisec(j[3]))

    tr = read(audio[start:end])

    result = model.transcribe(tr, fp16=False)

    f = open("Desktop/projects_python/tran_diar/tr_file.txt", "a")
    f.write(f'\n[ {j[1]} -- {j[3]} ] {j[6]} : {result["text"]}')
    f.close()

    del f
    del result
    del tr
    del j
    gc.collect()

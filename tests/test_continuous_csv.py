import numpy as np
import pandas as pd
import pytest
from ui.data import continuous_beats
from ml.preprocessing import normalize_beat

def test_continuous_segmentation_uses_original_signal(monkeypatch):
    signal=np.sin(np.arange(2000)/17)+np.arange(2000)/1000
    frame=pd.DataFrame({'MLII':signal,'sample':np.arange(2000)})
    monkeypatch.setattr('wfdb.processing.xqrs_detect',lambda *a,**k: np.array([50,300,900,1900]))
    raw,peaks,beats,skipped=continuous_beats(frame,'MLII',360,'sample')
    assert peaks.tolist()==[300,900] and skipped==2
    np.testing.assert_array_equal(beats[0],normalize_beat(signal[120:480]))
    assert beats.shape==(2,360)

def test_continuous_rejects_bad_recordings():
    frame=pd.DataFrame({'MLII':np.zeros(2000),'sample':np.arange(2000)})
    with pytest.raises(ValueError,match='360 Hz'): continuous_beats(frame,'MLII',250)
    with pytest.raises(ValueError,match='flat'): continuous_beats(frame,'MLII',360)
    with pytest.raises(ValueError,match='5 seconds'): continuous_beats(frame[:100],'MLII',360)
    frame.loc[10,'sample']=8
    with pytest.raises(ValueError,match='gaps'): continuous_beats(frame,'MLII',360,'sample')
    frame.loc[10,'MLII']=np.nan
    with pytest.raises(ValueError,match='missing'): continuous_beats(frame,'MLII',360)

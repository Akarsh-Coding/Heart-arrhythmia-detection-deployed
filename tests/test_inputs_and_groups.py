from io import StringIO
import numpy as np
import pandas as pd
import pytest
from ml import config
from ml.dataset import patient_id, stratified_sample_indices
from ml.train import training_folds
from ui.data import read_csv_frame, single_beat, batch_beats


def test_single_csv_requires_explicit_signal_column():
    frame = pd.DataFrame({'time': np.arange(360), 'ecg': np.sin(np.arange(360))})
    with pytest.raises(ValueError, match='Select exactly'):
        single_beat(frame)
    assert single_beat(frame, 'ecg').shape == (360,)
    with pytest.raises(ValueError):
        single_beat(pd.DataFrame(np.zeros((180, 2))))


@pytest.mark.parametrize('bad', [np.nan, np.inf, 'invalid'])
def test_malformed_cells_are_not_dropped(bad):
    frame = pd.DataFrame([list(range(360))], dtype=object)
    frame.iloc[0, 42] = bad
    with pytest.raises(ValueError):
        single_beat(frame)
    with pytest.raises(ValueError):
        batch_beats(frame)


def test_header_and_label_are_explicit():
    text = 'ecg\n' + '\n'.join(map(str, range(360)))
    assert single_beat(read_csv_frame(StringIO(text), True)).shape == (360,)
    with pytest.raises(ValueError):
        single_beat(read_csv_frame(StringIO(text)))
    frame = pd.DataFrame([list(range(360)) + ['N']])
    with pytest.raises(ValueError):
        batch_beats(frame)
    assert batch_beats(frame, label_column=360).shape == (1, 360)


def test_grouped_folds_keep_patients_together():
    y = np.tile(np.arange(5), 4)
    groups = np.repeat(['a', 'b', 'c', 'd'], 5)
    seen = []
    for train, val in training_folds(y, groups):
        assert not set(groups[train]) & set(groups[val])
        assert set(y[train]) == set(y[val]) == set(range(5))
        seen.extend(val)
    assert sorted(seen) == list(range(len(y)))
    with pytest.raises(ValueError, match='too few patients'):
        training_folds(y, np.repeat('same_patient', len(y)))
    assert patient_id('201') == patient_id('202')
    assert not {patient_id(r) for r in config.TRAIN_RECORDS} & {patient_id(r) for r in config.DS2_TEST_RECORDS}


def test_shap_sampling_covers_classes_without_duplicates():
    labels = np.repeat(np.arange(5), [100, 10, 5, 2, 1])
    indices = stratified_sample_indices(labels, 20)
    assert len(set(indices)) == 20
    assert set(labels[indices]) == set(range(5))
    np.testing.assert_array_equal(indices, stratified_sample_indices(labels, 20))
    with pytest.raises(ValueError):
        stratified_sample_indices(labels, 4)

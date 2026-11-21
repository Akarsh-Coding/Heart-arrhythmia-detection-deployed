import json
import numpy as np
import pytest
import tensorflow as tf
from ml import config
from ml import train as run


def tiny_model():
    model = tf.keras.Sequential([tf.keras.layers.Input((360, 1)),
                                tf.keras.layers.GlobalAveragePooling1D(),
                                tf.keras.layers.Dense(5, activation='softmax')])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def samples():
    X = np.random.default_rng(42).normal(size=(10,360,1)).astype('float32')
    y = np.tile(np.arange(5),2)
    return X,y,tf.keras.utils.to_categorical(y,5)


def test_schedule_uses_fold_histories():
    folds=[{'best_epoch':3,'history':[{'learning_rate':x} for x in [0.001,0.0005,0.00025]]},
           {'best_epoch':2,'history':[{'learning_rate':x} for x in [0.001,0.001]]}]
    assert run.final_schedule(folds)==[0.001,0.00075]


def test_checkpoint_resume_and_best_selection(tmp_path):
    X,_,y=samples()
    first,state=run.fit_stage(tmp_path,X,y,1,validation=(X,y),model_factory=tiny_model,batch_size=5)
    iterations=int(first.optimizer.iterations.numpy())
    expected=first(X,training=False).numpy()
    loaded=tf.keras.models.load_model(tmp_path/state['checkpoint'])
    np.testing.assert_allclose(loaded(X,training=False),expected,atol=1e-7)
    assert int(loaded.optimizer.iterations.numpy())==iterations
    _,state=run.fit_stage(tmp_path,X,y,2,validation=(X,y),model_factory=tiny_model,batch_size=5)
    latest=tf.keras.models.load_model(tmp_path/state['checkpoint'])
    assert int(latest.optimizer.iterations.numpy())==iterations*2
    assert len(state['history'])==2
    assert (tmp_path/state['best_checkpoint']).exists()


def test_pipeline_reports_and_resume(tmp_path,monkeypatch):
    from ml import evaluate
    from ml.artifacts import validate_metadata
    X,y,_=samples()
    monkeypatch.setattr(config,'MODEL_DIR',tmp_path/'models')
    monkeypatch.setattr(config,'REPORTS_DIR',tmp_path/'reports')
    monkeypatch.setattr(config,'CV_FOLDS',2)
    monkeypatch.setattr(config,'EPOCHS',1)
    monkeypatch.setattr(run,'load_split',lambda _, return_groups=False: (X,y,np.repeat(['a','b'],5)) if return_groups else (X,y))
    monkeypatch.setattr(evaluate,'load_split',lambda _: (X,y))
    monkeypatch.setattr(evaluate,'REPORTS_DIR',tmp_path/'reports')
    original=run.fit_stage
    def small_stage(*args,**kwargs):
        return original(*args,**kwargs,model_factory=tiny_model,batch_size=5)
    monkeypatch.setattr(run,'fit_stage',small_stage)
    run.run_training()
    validate_metadata(config.MODEL_DIR/'best_mshan_model.keras')
    assert (config.REPORTS_DIR/'confusion_matrix.png').exists()
    assert (config.REPORTS_DIR/'roc_curves.png').exists()
    metrics=json.loads((config.REPORTS_DIR/'evaluation_metrics.json').read_text())
    assert np.isfinite(metrics['Macro_AUC'])
    state_path=config.MODEL_DIR/'training'/'final'/'state.json'
    before=state_path.read_text()
    run.run_training()
    assert state_path.read_text()==before
    monkeypatch.setattr(config,'INITIAL_LR',0.01)
    with pytest.raises(ValueError,match='changed'):
        run.run_training()

def test_interrupted_epoch_restarts_from_committed_checkpoint(tmp_path,monkeypatch):
    X,_,y=samples()
    _,first=run.fit_stage(tmp_path,X,y,1,schedule=[0.001],model_factory=tiny_model,batch_size=5)
    original=run.write_json
    def interrupt(path,value):
        raise RuntimeError('simulated interruption before epoch state commit')
    monkeypatch.setattr(run,'write_json',interrupt)
    with pytest.raises(RuntimeError,match='simulated'):
        run.fit_stage(tmp_path,X,y,2,schedule=[0.001,0.0005],model_factory=tiny_model,batch_size=5)
    assert len(json.loads((tmp_path/'state.json').read_text())['history'])==1
    monkeypatch.setattr(run,'write_json',original)
    model,state=run.fit_stage(tmp_path,X,y,2,schedule=[0.001,0.0005],model_factory=tiny_model,batch_size=5)
    assert len(state['history'])==2
    assert int(model.optimizer.iterations.numpy())==4
    assert state['history'][1]['learning_rate']==pytest.approx(0.0005)

def test_plateau_and_early_stop_state_survive_resume(tmp_path,monkeypatch):
    from types import SimpleNamespace
    X,_,y=samples()
    def fixed_loss(self,*args,**kwargs):
        return SimpleNamespace(history={'loss':[1.0],'accuracy':[0.2],
                                        'val_loss':[1.0],'val_accuracy':[0.2]})
    monkeypatch.setattr(tf.keras.Model,'fit',fixed_loss)
    def built_model():
        model=tiny_model()
        model.optimizer.build(model.trainable_variables)
        return model
    _,state=run.fit_stage(tmp_path,X,y,3,validation=(X,y),model_factory=built_model)
    assert state['early_wait']==2
    assert state['lr_wait']==0
    _,state=run.fit_stage(tmp_path,X,y,10,validation=(X,y),model_factory=built_model)
    assert state['stopped'] and len(state['history'])==7
    assert state['best_epoch']==1
    assert state['history'][3]['learning_rate']==pytest.approx(0.0005)
    assert state['history'][5]['learning_rate']==pytest.approx(0.00025)

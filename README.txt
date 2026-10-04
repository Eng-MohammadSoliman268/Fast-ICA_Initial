FastICA Core v1.0
==================

Files
-----
1) fastica_core.py
   Final reusable FastICA core.

2) validate_fastica_vs_mne.py
   Validation benchmark against MNE FastICA on the same preprocessed data.

3) test_fastica_vs_mne_colab.ipynb
   Minimal Colab/Jupyter example.

Core design
-----------
The FastICA core intentionally performs only:
- ICA fitting / separation
- transform to IC space
- inverse transform / reconstruction
- diagnostics
- save/load fitted transform

It intentionally does NOT perform:
- artifact IC detection
- artifact classification
- artifact removal
- EEG filtering
- bad-channel handling
- interpolation
- re-referencing

Recommended EEG workflow
------------------------
EEG acquisition
    -> preprocessing
    -> FastICAFromScratch
    -> independent components
    -> (future) artifact IC detector
    -> (future) artifact rejection
    -> inverse_transform for reconstruction

Quick use
---------
from fastica_core import FastICAFromScratch

ica = FastICAFromScratch(
    n_components=None,
    max_iter=2000,
    tol=1e-6,
    fun="logcosh",
    alpha=1.0,
    max_train_samples=50000,
    random_state=42,
)

S = ica.fit_transform(X)
X_hat = ica.inverse_transform(S)

print(ica.get_summary())
print(ica.reconstruction_error(X))

Validate against MNE
--------------------
from validate_fastica_vs_mne import compare_fastica_with_mne

result = compare_fastica_with_mne(
    X=X_ica,
    sfreq=sfreq,
    ch_names=good_channels,
    max_train_samples=50000,
    max_iter=2000,
    tol=1e-6,
    random_state=42,
)

print(result["summary"])
print(result["mapping"].head())

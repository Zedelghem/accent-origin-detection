# accent-origin-detection
A minimal implementation of the accent-origin classifier from the UCLA
M214A Project to detect contemporary and historical accents from
several US cities based on the CORAAL database (cf. https://oraal.github.io/coraal).
This implementation uses only basic audio features and XGBoost.
No embeddings, no neural networks, pure decision-tree extravaganza.

The training pipeline is available in the `main.ipynb` file. Intermediary files 
and plots available in the subfolder.

The helper functions assume the following directory structure:

```
- project_data/
    - train/
        - sample1.wav/
        - sample2.wav/
    - test_clean/
        - sample1.wav/
        - sample2.wav/
    - test_noisy/
        - sample1.wav/
        - sample2.wav/
- features/
```

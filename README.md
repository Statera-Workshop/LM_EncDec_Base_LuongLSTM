# LM_EncDec_Base_LuongLSTM

Encoder-decoder translation model ported from legacy TF2 code C5 to PyTorch (cell=lstm, attention=general).

## Status

- Verification level: **Benchmarked (LM-03, one run)**

## Changes from the legacy code

- PyTorch port; beam search reimplemented (TF Addons is no longer maintained)
- Vocabulary built from the training split only (legacy built it from all pairs)
- "what's" expanded to "what is" (legacy wrote "that is")

## LM-03 setting

English -> French, 5-11 words, 12,800 pairs (seed 1234), 80/20 split, embedding 128, hidden 1024, batch 256, 20 epochs, Adam 1e-3. Small data: compare structures, not absolute BLEU.

## Results

- Parameters: 26.2 M
- Test perplexity 37.1, next-word accuracy 0.530
- BLEU-4 greedy (whole test split): 16.6
- BLEU-4 beam width 4 on 500 pairs: 18.6 (greedy on the same pairs 16.7)
- Examples:
  - `this winter has been mild .` -> `cette annee est ete court .` (reference `cet hiver a ete doux .`)
  - `they will be here at three .` -> `ils pourraient tous ici pendant trois heures .` (reference `ils seront ici a trois heures .`)
  - `i once lived in rome .` -> `il me faut une autre d'amis .` (reference `j'ai vecu un temps a rome .`)

## Run

```bash
pip install torch==2.5.1 einops numpy
# put fra-eng.zip from https://www.manythings.org/anki/ into data/
python run.py lm03 s2s_luong_lstm
```

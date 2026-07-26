"""Reproduction script for the zim_boost contribution to supercon_chem_Tc.

Pipeline (all selection done by 5-fold CV on the official train split; the official test
split is predicted exactly once at the end):

  1. Parse formulas (with cleanup of SuperCon formula artifacts like trailing charges).
  2. Featurize: matminer composition featurizers (Magpie ElementProperty, Stoichiometry,
     ValenceOrbital, IonProperty, ElementFraction, TMetalFraction, BandCenter) plus
     superconductivity-physics descriptors (charge-balanced cuprate hole doping p and the
     Presland-Tallon dome 1-82.6(p-0.16)^2, Matthias valence-electron-count rules,
     isotope-effect <M^-1/2> mass proxies, block fractions, family indicators).
  3. Zero-inflated median architecture:
        pred(x) = 0                      if P(Tc=0 | x) >= threshold
                = mean_k R_k(x)          otherwise
     motivated by 24.4% of labels being exactly 0 and MAE being minimised by the
     conditional median: for the zero-inflated mixture the median is exactly 0 whenever
     P(Tc=0|x) >= 1/2 (threshold tuned by CV around that value).
     P(Tc=0) comes from LightGBM classifiers (optionally averaged with a composition-space
     kNN classifier); R_k are positive-part-trained regressors (LightGBM L1/L2, random
     forest, composition-space kNN over the element-fraction simplex).
  4. Refit chosen components on the full official train split, predict the official test
     split once, write AI-SinglePropertyPrediction-Tc-supercon_chem-test-mae.csv.zip.

Full source: featurize.py / physfeat.py / add_phys.py / components.py / knn_comp.py /
combine.py / submit.py in the accompanying repository. Deterministic seeds (42 for CV,
0 for RF) throughout.

Data: jarvis_tools figshare `supercon_chem` (16,414 rows) with the official benchmark split
jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/supercon_chem_Tc.json.zip
(train 13,131 / test 3,283).
"""
print(__doc__)

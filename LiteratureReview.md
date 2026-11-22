# Literature Review: Traffic Sign Detection on LISA

Classical traffic-sign systems are usually **two-stage**: (1) **region of interest (ROI) extraction** on the full frame, then (2) **classification** of each crop. On the LISA Traffic Sign dataset, the main classical ROI methods are **HOG**, **integral channel features (ICF)**, and **aggregate channel features (ACF)**. After about 2015 the published state of the art is deep learning; those papers matter here mainly for **how they preprocess LISA**, not for the detector architecture.

This review compares those ROI features, explains why ICF/ACF use **boosting** rather than an SVM, and summarizes preprocessing choices in later LISA work.

---

## 1. The LISA Traffic Sign dataset

LISA-TS (Laboratory for Intelligent and Safe Automobiles, UCSD) is a **United States** traffic-sign set: about **6,610 frames**, **7,855** annotated signs, **47** types [Møgelmose et al., 2012]. Sign crops are small (**6×6 to ~167×168** pixels). Frames mix **color and grayscale**; resolutions run about **640×480 to 1024×522**. A 2015 extension adds HD video tracks meant for **tracking**, not only single-frame detection [Møgelmose et al., 2014; 2015].

LISA is harder than the German Traffic Sign Detection Benchmark (GTSDB) for two design reasons:

- Many US signs are **white rectangles** (speed limits and similar) with little color or distinctive outline.
- Warning signs are **yellow diamonds**, not European red triangles / blue circles.

A method that is near-perfect on GTSDB therefore need not transfer [Møgelmose et al., 2015].

**Protocol note.** Consecutive LISA frames are video tracks. Random train/test splits **leak**. Papers that treat classification seriously split **by track** [Bodington et al., 2014]. Detection papers use PASCAL overlap: a box is a true positive if IoU with ground truth is **> 0.5**.

---

## 2. Classical two-stage pipeline

```text
frame
  → ROI extractor (sliding window or color/MSER proposals)
      → candidate boxes
  → crop / resize
  → classifier (HOG+SVM/LDA, or a CNN on the crop)
  → optional tracking across the video track
```

Color/MSER/Haar-cascade proposals appear in student OpenCV pipelines. The methods that actually define classical performance on US signs are **sliding-window channel detectors** (ICF/ACF) and **HOG** (as a window descriptor or as a second-stage classifier).

Tracking on LISA video tracks **suppresses almost all false positives** in the 2014 study; any practical detector should include it [Møgelmose et al., 2014].

---

## 3. ROI features: HOG vs ICF vs ACF

### 3.1 Histogram of oriented gradients (HOG)

HOG [Dalal and Triggs, 2005] builds **unsigned gradient histograms** in cells, normalizes them in overlapping blocks, and feeds a **linear SVM**. It is a **shape** descriptor. It is usually computed on **grayscale** (or on the strongest-gradient color channel).

On LISA that is a real limitation: stop signs, yellow diamonds, and red/blue regulatory signs carry **color**. HOG still does relatively well on **speed-limit** signs, which are black-on-white rectangles [see §3.4].

HOG is used in two different places in the literature, which should not be mixed:

| Role | Typical setup | What is measured |
| --- | --- | --- |
| **ROI / detector** | Sliding window + linear SVM (or LDA) on every scale | Precision–recall / AUC on full images |
| **Classifier** | Crop already given; 32×32 grayscale HOG → LDA/SVM | Classification accuracy on crops |

Bodington et al. [2014] are **classification only**: crop to a square around the sign, grayscale, **32×32**, eight frequent classes, **track-aware split**. HOG+LDA already reached about **98%** on that subset.

More and Bos [2020] compare **HOG+linear SVM vs YOLO** as detectors on LISA: YOLO is more accurate; HOG is faster on an unaccelerated CPU.

### 3.2 Integral channel features (ICF)

ICF / ChnFtrs [Dollár et al., 2009] builds **10 registered channels**, then scores Haar-like features with **integral images**:

1. **L, U, V** color
2. **Gradient magnitude**
3. **Six oriented-gradient** maps

A feature is the **difference of two rectangle sums** on one channel (first-order Haar). Higher-order Haar barely helps. An **AdaBoost** forest of **depth-2 trees** is trained on a random pool of such features and run as a **sliding window**. Mathias et al. [2013] brought this detector to GTSDB (team VISICS) and reached near-perfect AUC on European superclasses.

ICF is a **strict superset of HOG-like information**: it has oriented gradients **and** color **and** local contrast (Haar), with a cascade instead of a dense SVM.

### 3.3 Aggregate channel features (ACF)

ACF [Dollár et al., 2014] keeps the **same 10 channels** and the **same boosted forest**. It **replaces Haar** with **pooled (aggregated) pixel blocks** at several scales. A feature is essentially a **lookup of a cell sum**, not a two-rectangle difference.

A related 2014 result is **fast feature pyramids**: nearby scales are approximated instead of recomputing channels at every scale. That is why ACF is the practical real-time variant. ACF was proposed as a **faster ICF** and is **sometimes more accurate**.

| | **HOG** | **ICF** | **ACF** |
| --- | --- | --- | --- |
| Channels | Gradients (usually gray) | LUV + \|∇\| + 6 orientations | Same as ICF |
| Features | Cell/block histograms, L2-norm | Haar on **integral images** | **Block sums / cell lookups** |
| Classifier | Linear SVM (typical) | AdaBoost, depth-2 trees | Same as ICF |
| Designed for | Dense descriptor + SVM | Weak-feature cascade | Faster ICF + scale pyramid |
| Color | Weak unless added by hand | Explicit LUV | Explicit LUV |

**Intuition.** HOG asks “what is the distribution of edge directions?” ICF asks “is this patch darker/redder/edgier than that neighboring patch?” ACF asks “what is the pooled value of this cell on this channel?”

### 3.4 Detection numbers

There is **no published head-to-head HOG vs ICF table on LISA**. The fair ROI comparison is GTSDB; LISA reports **ICF vs ACF**.

**GTSDB** (AUC of the precision–recall curve, IoU ≥ 0.5) [Mathias et al., 2013; [GTSDB results](https://benchmark.ini.rub.de/gtsdb_results_all.html)]:

| Superclass | ICF (VISICS / Mathias “competition”) | HOG+SVM (LITS1) | HOG+LDA+SVM (wgy@HIT501) |
| --- | ---: | ---: | ---: |
| Prohibitory (red circle) | **100%** | ~98–99% | **100%** |
| Danger (red triangle) | **100%** | ~97% | **100%** |
| Mandatory (blue circle) | **96.98%** | **92%** | **99.91%** |

Vanilla **HOG+SVM trails ICF by a few AUC points** (about **+5** on mandatory). A **cascaded, well-tuned HOG** (wgy) **matches ICF**. Dollár et al. [2009] report the same pattern for pedestrians: ICF **86%** recall at 1 FP/image vs HOG **77%** on INRIA full images.

**Extended LISA-TS** (3346 train / 2321 test frames) [Møgelmose et al., 2015]:

| Superclass | ACF AUC | ICF AUC |
| --- | ---: | ---: |
| Diamond warning | **98.98** | lower (ACF “significantly” better on US signs) |
| Stop | **96.11** | slightly below ACF |
| No-turn | **96.17** | slightly below ACF |
| Speed limit (white rectangle) | **84.31** | **87.91** (ICF wins) |

On European signs ICF and ACF are nearly tied (high-90s AUC). On **salient US signs, ACF wins**. On **speed limits, ICF wins**. Channel ablation in the same paper: **shape ≫ color**; **color-only ICF fails completely** on speed limits; ACF leans more on color, which explains the crossover.

**Implication for this project.** Use **ACF as the default ROI extractor**. It is faster than ICF, stronger on colored/shaped signs, and only behind ICF on the colorless rectangular superclass where HOG is also competitive. If speed-limit detection is a required class, keep an **ICF or HOG** variant in mind rather than expecting ACF to close that gap.

---

## 4. Why ICF and ACF use boosting, not an SVM

Boosting is a **detector** choice, not a claim that SVMs are weak classifiers. HOG + **linear SVM** is the competing architecture [Dalal and Triggs, 2005]. ICF/ACF follow **Viola–Jones**: a cafeteria of **cheap weak features** plus a **sliding window** over 10⁴–10⁶ boxes per image [Dollár et al., 2009; Viola and Jones, 2001].

**What boosting does here**

1. **Feature selection.** AdaBoost picks, each round, the Haar or cell that most reduces error. Unused features are never computed at test time. Depth-2 trees also form tiny conjunctions (“U-channel high **and** gradient magnitude high”).
2. **Cascade / early exit.** After a few trees, almost every background window already has a terrible score and is **rejected**. Detection is 1 positive vs ~10⁵ negatives per image; that is the setting boosting was built for.
3. **Weak features are allowed.** A single Haar on one channel is a terrible classifier. Boosting assumes that and adds 1000–2000 weak learners.

Dollár et al. keep a “standard boosting approach” and spend design effort on **channels**, not on a new learner. Depth-2 trees beat stumps; the **choice of boosting algorithm barely matters**.

**What goes wrong if you drop in an SVM**

| | Boosting (ICF/ACF) | Linear SVM (typical HOG) |
| --- | --- | --- |
| Input | Pool of weak Haar/cells | One dense HOG per window |
| Feature selection | Yes, during boosting | No (all bins used) |
| Nonlinearity | Depth-2 trees | Must live in the descriptor |
| Test cost | First trees reject most windows | Full HOG + full \(w^\top x\) always |
| Fits | Heterogeneous channels, scanning | Fixed descriptor, fewer windows |

A linear SVM needs a **fixed dense** \(x\) and a full dot product **on every window**. For ICF, \(x\) would be “every Haar on 10 channels” (thousands to millions of dimensions). You either compute all of them always, or you select features by hand—the job boosting already does. Kernel SVM can match the trees on accuracy and then **cannot** run on a dense scale pyramid.

**SVM is the right tool for stage 2.** Once a crop exists (32×32 sign), HOG+SVM or HOG+LDA is appropriate [Bodington et al., 2014]. The mismatch is **SVM as the ICF/ACF window scorer**, not SVM as a sign classifier.

---

## 5. Deep learning on LISA, with emphasis on preprocessing

After 2015 most LISA papers are detectors (YOLO, SSD, Faster/Cascade R-CNN, CenterNet). They rarely train on raw mixed LISA as-is. Recurring **pre-network** steps:

| Step | Why it shows up on LISA |
| --- | --- |
| Drop grayscale frames | Color models; superclasses use color |
| Keep only frequent classes (8–16) or merge to 3 superclasses | 47 classes, long tail |
| Crop signs **or** letterbox whole frames | Tiny objects vs fixed input size |
| Resize (32×32, 256×256, 400×260, 640×640) | Fixed tensor |
| Contrast / CLAHE (`imadjust`, `histeq`, `adapthisteq`) | Illumination in driving video |
| Track-aware train/test split | Consecutive frames leak |
| Light aug (rotate, noise) or GAN night | Class imbalance / domain shift |

Representative papers:

- **Vázquez-Santiago et al. [2021]**: color frames only (drop ~60% grayscale), collapse 47 types to **warning / prohibitory / mandatory**, balance to 900 images (“Tiny LISA”), SSD+FPN (MobileNet-v1 / ResNet-50).
- **He et al. [2021]**: ROI blocks, pedestrian-style HOG (64×128, 8×8 cells), contrast enhancement, then CapsNet. Useful for the **contrast + HOG** recipe; the learning claims should be treated cautiously.
- **Mishra et al. [2022]**: Faster R-CNN, SSD, YOLOv3, RetinaNet, CenterNet, Cascade R-CNN on LISA-TS (and other sets). Fine labels merged to **meta-classes**. **CenterNet / Cascade R-CNN** do best; the preprocess is standard detector resize, not custom CV filters.
- **Roychowdhury et al. [2019]**: LISA-TS + extension resized to **256×256**, 80/20 split, GAN+RL night augmentation.
- **YOLOv5 + C2f + EIoU [2025]**: keep the **16 most frequent** LISA types; usual YOLO letterbox to 640, RGB, /255.
- **More and Bos [2020]**: HOG+SVM vs YOLO on LISA (accuracy vs CPU speed).

**GLARE** [2022] extends the US-sign setting to **sun glare**. Temel et al. [2019] study detector reliability under weather and noise and discuss LISA tracks.

For a **classical** project, ignore most of the DL architecture and copy the **LISA hygiene**: consistent color vs gray, drop rare classes or use superclasses, letterbox or crop to a fixed size, enhance contrast **before** features, and split **by video track**.

---

## 6. Design takeaways

1. **ROI:** **ACF** is the default classical detector (speed + accuracy on colored/shaped signs). **ICF** (or HOG) if the target set is dominated by **white rectangular** speed limits.
2. **Classifier on crops:** **HOG + LDA/SVM** is enough for a small, frequent-class subset; boosting is not required once the window is given.
3. **Do not put a linear SVM on the ICF/ACF Haar pool** unless the goal is to rebuild a HOG-style dense descriptor. The forest **is** the feature selector and the cascade.
4. **Always** track-aware splits, PASCAL 0.5 IoU for detection, and (if video is available) **tracking** to kill false positives.
5. Deep learning is the published SOTA; on LISA its practical lesson is **preprocessing and class subsetting**, not a requirement to abandon ACF for a course-scale classical system.

---

## References

Bodington, D., Greenstein, E., and Hu, M. (2014). *Implementing machine learning algorithms on GPUs for real-time traffic sign classification*. CS229, Stanford. [PDF](https://cs229.stanford.edu/proj2014/Dashiell%20Bodington,%20Eric%20Greenstein,%20Matthew%20Hu,%20Implementing%20Machine%20Learning%20Algorithms%20on%20GPUs%20for%20Real-Time%20Traffic%20Sign%20Classification.pdf)

Dalal, N. and Triggs, B. (2005). Histograms of oriented gradients for human detection. *CVPR*.

Dollár, P., Tu, Z., Perona, P., and Belongie, S. (2009). Integral channel features. *BMVC*. [PDF](https://vision.ucsd.edu/sites/default/files/publications/pdfs/Integral%20Channel%20Features.pdf)

Dollár, P., Appel, R., Belongie, S., and Perona, P. (2014). Fast feature pyramids for object detection. *IEEE TPAMI* (ACF / feature pyramids).

He, S., Chen, L., Zhang, S., Guo, Z., Sun, P., Liu, H., and Liu, H. (2021). Automatic recognition of traffic signs based on visual inspection. *IEEE Access*. [doi](https://doi.org/10.1109/access.2021.3059052)

Mathias, M., Timofte, R., Benenson, R., and Van Gool, L. (2013). Traffic sign recognition — How far are we from the solution? *IJCNN*. [PDF](https://btsd.ethz.ch/shareddata/publications/Mathias-IJCNN-2013.pdf)

Mishra, A., et al. (2022). Evaluating and bench-marking object detection models for traffic sign and traffic light datasets. *ACCV Workshops*. [PDF](https://openaccess.thecvf.com/content/ACCV2022W/DLSOD/papers/Mishra_Evaluating_and_Bench-marking_Object_Detection_Models_for_Traffic_Sign_and_Traffic_Light_Datasets_ACCVW_2022_paper.pdf)

Møgelmose, A., Trivedi, M. M., and Moeslund, T. B. (2012). Vision-based traffic sign detection and analysis for intelligent driver assistance systems: Perspectives and survey. *IEEE T-ITS*. [doi](https://doi.org/10.1109/tits.2012.2209421)

Møgelmose, A., Liu, D., and Trivedi, M. M. (2014). Traffic sign detection for U.S. roads: Remaining challenges and a case for tracking. *IEEE ITSC*. [doi](https://doi.org/10.1109/itsc.2014.6957882)

Møgelmose, A., Liu, D., and Trivedi, M. M. (2015). Detection of U.S. traffic signs. *IEEE T-ITS*. [doi](https://doi.org/10.1109/tits.2015.2433019) · [PDF](https://vbn.aau.dk/ws/portalfiles/portal/210185909/signsITSTrans2015.pdf)

More, S. and Bos, J. (2020). Comparing machine learning and neural network-based approaches for sign detection and classification in autonomous vehicles. *Proc. SPIE* 11415. [doi](https://doi.org/10.1117/12.2558966)

Roychowdhury, S., et al. (2019). Automated augmentation with reinforcement learning and GANs for robust identification of traffic signs using front camera images. [arXiv:1911.06486](https://arxiv.org/abs/1911.06486)

Temel, D., Chen, M.-H., and AlRegib, G. (2019). Challenging environments for traffic sign detection: Reliability assessment under inclement conditions. [arXiv:1902.06857](https://arxiv.org/abs/1902.06857)

Vázquez-Santiago, D. I., et al. (2021). Evaluation method of deep learning-based embedded systems for traffic sign detection. *IEEE Access*. [doi](https://doi.org/10.1109/access.2021.3097969)

Viola, P. and Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. *CVPR*.

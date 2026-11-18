<div align="center">

# How many indicators does development monitoring need?

### A minimal, stable development index and three policy archetypes

**Group 16** · Chen Wanchuan · Lu Yuqing · Yuen Lok Yan

DSAI5102 Principles of Data Science — Mini-Project, Format 1 (Data Application and Practice)

</div>

## Abstract

Development monitoring is indicator-rich and data-poor at the same time: the frameworks used to target policy run to dozens or hundreds of indicators, while the countries that most need targeting report the fewest of them. This paper asks how much of that apparatus is actually necessary. Using 28 World Development Indicators for 198 economies over 2005–2022, we compress the country profile to a single development axis, then ask how few indicators reproduce it and how many *kinds* of country it takes to describe the world. Four results stand out. First, one principal component explains 47.9% of the variance (95% bootstrap CI 44.4–51.6%), and the ordering it defines is insensitive to its inputs: ρ = 0.9986 or better after deleting any single indicator, 0.9974 or better across country bootstrap resamples, and 0.9826 or better across four non-overlapping time windows. Second, the answer to "how few" depends on the criterion, and the two criteria disagree almost completely: their shortlists overlap in 2 of 22 indicators (Jaccard 0.14). A label-blind sparse-PCA shortlist of 12 reproduces the full ranking at ρ = 0.9935 and predicts income groups as well as all 28 indicators (macro-F1 0.821 against 0.834). A label-aware greedy shortlist needs only 3 indicators and scores higher (0.866), but that advantage does not survive an honest protocol: when selection is moved inside the cross-validation fold, its balanced accuracy falls to 0.803. Third, the country cloud supports three stable archetypes — advanced, mature middle, emerging frontier — with a bootstrap ARI of 0.953 and a distinguishable profile each: digital infrastructure is the middle type's gap, electrification, schooling and survival are the frontier's. Fourth, the index recovers documented development episodes it was never told about. Across two non-overlapping windows the ranking persists at ρ = 0.986, while China gains 32 rank places and Venezuela loses 34. The practical implication is a division of labour rather than a reduction: three near-complete series will screen a country, twelve will place it, and the rest buys diagnosis rather than ordering.

**Keywords:** composite index, principal component analysis, sparse PCA, feature selection, clustering stability, development monitoring, World Development Indicators

## 1. Introduction

### 1.1 The problem

Suppose a ministry has to decide where to send a development programme next year, and it has a budget for five countries. What does it need to know? In practice it is offered a dashboard: gross domestic product, life expectancy, schooling, electrification, internet penetration, air quality, forest cover, maternal mortality, and a hundred more series, most of them updated annually, several of them revised retroactively. The frameworks that organise this abundance — the Sustainable Development Goals with their 231 unique indicators [1], the Human Development Index [2], the Multidimensional Poverty Index [3] — are genuinely useful, but they were built by *adding* dimensions, on the argument that development is multidimensional and any single number hides something.

That argument is correct, and it creates a specific practical difficulty. Composite indices in common use aggregate ten to a few hundred series with weights that are chosen, not estimated: equal weights in the HDI, arbitrary or expert weights in most others. The composite indicator literature has long warned that such weights are a modelling decision with consequences as large as the data themselves [4, 5], and that the ranking of countries is sensitive to them [6]. A second difficulty is symmetric and less discussed: the countries at the bottom of the ranking are precisely the ones whose statistical systems produce the least. Missingness in development data is not random — it is concentrated in exactly the places where the monitoring is supposed to guide action. Any index that needs 28 series to place a country is an index that works best where it is needed least.

### 1.2 The question

This paper therefore asks the complementary question to the usual one. Instead of *how do we add more dimensions*, we ask **how few do we need**, and we ask it at three levels of increasing demand:

- **RQ1 — Does development have one dominant axis, and is that axis stable?** If a single component carries roughly half the variance of a 28-indicator profile, and if the country ranking it induces survives changes of indicator set and time window, then the dimensionality of the problem is not 28.
- **RQ2 — What is the smallest indicator set that reproduces the index and still does the job?** "Does the job" means two things: reproducing the ranking (fidelity), and predicting the income group a country belongs to (usefulness). These are different criteria and, we will show, they select different sets.
- **RQ3 — How many kinds of country are there, and who is moving?** A ranking is a line; a policy typology is a partition. We ask how many clusters the country cloud supports when stability, not just separation, is the selection criterion, and whether the resulting types move in distinguishable ways over time.

### 1.3 Why this is a data-science problem and not a bookkeeping exercise

Each of the three questions is a place where a careless analysis produces a confident and wrong answer, and the discipline that separates the two is the substance of the paper.

For RQ1, the naive move is to read the first principal component's share of variance as the "importance of development". That number is a property of the indicator set, not of the world: add five collinear macroeconomic series and the share rises; add five noisy environmental series and it falls. What is worth reporting is not the share but the *stability of the ordering* under perturbation, which is a resampling question.

For RQ2, the naive move is to select indicators by their correlation with the outcome and then report the cross-validated accuracy of the selected set. This is selection bias of a well-known kind [7]: the same data inform both the choice and the evaluation, so the reported accuracy is optimistic, sometimes spectacularly so. Two things can be done. One is to select *without* looking at the label, which makes the accuracy estimate honest because the selection cannot have used the outcome. The other is nested cross-validation, in which selection happens inside each training fold. We do both and report the gap between them, because the gap is itself a finding.

For RQ3, the naive move is to choose the number of clusters by silhouette or elbow, both of which measure separation *in the sample* and say nothing about whether the partition would reappear in a different sample of countries. Clustering has no ground truth to validate against, so stability under resampling is the only evidence available [8, 9]. We use it as the primary criterion and report silhouette only as a secondary one.

### 1.4 Contributions

1. **A dual-criterion answer to "how few indicators".** We separate *fidelity* — a shortlist that a label-blind sparse-PCA procedure can defend — from *usefulness* — a shortlist optimised to predict the income group — and we show that 12 indicators suffice for the first purpose, that 3 appear to suffice for the second, and that roughly half of that three-indicator advantage is selection bias rather than signal. Reporting only the second would overstate what development data can be compressed to; reporting only the first would understate it.
2. **An honest performance curve.** We report accuracy against shortlist size twice: once from a procedure that selects on the full sample, whose curve peaks misleadingly at three indicators (0.849 balanced accuracy), and once from nested cross-validation in which selection happens inside each training fold, where the same k reaches 0.803 and the curve peaks near k = 9–14 instead. The difference quantifies how much of the apparent gain is selection bias rather than signal.
3. **Stability-first typology.** The number of archetypes is chosen by bootstrap-adjusted Rand index rather than silhouette, which changes the answer: k = 2 has the best silhouette and is statistically tied with k = 3 on stability, so the choice between them is made on the content of the partition rather than on separation, and k ≥ 4 is rejected by stability itself.
4. **External validation through real episodes.** A stability statistic is reassuring but abstract; the index is also checked against development history it was not told about. Venezuela (−34 rank places), Syria (−22), Libya (−21) and Yemen (−19) fall; China (+32), Georgia (+26), Costa Rica (+24) and Peru (+20) rise. A noisy index does not locate a crisis.
5. **A reproducible, low-cost pipeline.** The entire study — data retrieval, imputation, index, selection, classification, clustering, bootstrap stability, figures and tables — runs in under fifteen minutes on a laptop CPU, with no GPU and no proprietary data. The code and the exact intermediate artefacts are released with the paper.

## 2. Related work

**Composite indicators.** The construction of composite indices is codified in the OECD/JRC handbook [4], whose central point is that every step — indicator selection, imputation, normalisation, weighting, aggregation — is a choice with distributional consequences. Saisana et al. [6] formalise this as uncertainty and sensitivity analysis and show that country ranks in well-known indices can move by many places under defensible alternative specifications. Our RQ1 is essentially a sensitivity analysis of one such specification, with the twist that we are looking for the *floor* of the sensitivity rather than the size of the error.

**Weighting and dimension reduction.** Three families of weighting coexist: normative (equal weights, as in the HDI [2]), participatory (expert or budget-allocation weights), and data-driven. Principal component analysis [10] is the standard member of the third family, and it has been used for development indices since the 1970s; its weakness is that each component is a dense combination of all inputs, which makes the index hard to *explain* to the analyst who has to defend the weights. Sparse PCA [11] addresses exactly this by adding an ℓ<sub>1</sub> penalty that drives loadings to zero, producing components loading on a handful of interpretable variables. We use sparse PCA in two roles: as a shortlist generator (RQ2, criterion A) and as a check on whether the dense index is an artefact of many small contributions.

**Variable selection.** Wrapper methods [12, 13] evaluate subsets by the performance of a downstream model and are the natural tool for RQ2's usefulness criterion. Their known hazard is that the same validation data drive both selection and assessment [7]; the standard remedy, nested cross-validation, is used here and its effect is measured rather than assumed.

**Cluster validation.** Internal indices (silhouette [14], within-cluster sum of squares) measure compactness and separation. External validation requires labels, which a typology does not have. The accepted third route is stability: perturb the data and ask whether the partition reappears [8, 9, 15]. Bootstrap-adjusted Rand indices [16] are the usual currency, and that is what we report.

**Why these five models, and who else uses them this way.** Table&nbsp;1 states what each model is for, what it buys and what it costs. Two constraints fixed the choice. First, the index must be defensible to an audience with no labels: any step that learns from the income groups would make the stability results circular, so the index and the shortlist generator are both unsupervised. Second, every claim has to be reproducible on a laptop, so nothing here needs a GPU. The models that were *not* used are as informative as the ones that were: factor analysis with rotation (the rotation is not identified by the data), autoencoder embeddings (expressive, but not identifiable and not interpretable), hierarchical or PAM clustering (the dendrogram invites a tree reading the data do not support), and heavily tuned gradient boosting (n = 198 is too small to tune honestly; we keep the default-pressure HistGradientBoosting only as a third, non-linear check).

::: {.tmodels}
::: {.tmodels}
::: {.tmodels}
| Model (role) | What it buys | What it costs | Used this way elsewhere |
|:--|:--|:--|:--|
| PCA (M1, index) | deterministic, unsupervised, variance-optimal; exactly reproducible | linear; variance is not importance; dense weights are hard to defend | PCA wealth indices in DHS and national surveys [17, 18]; a data-driven weighting option [4] |
| Sparse PCA (M2, blind shortlist) | components loading on a few series, so the shortlist can be named; no labels, no leakage | non-convex; the penalty must be chosen (knee here); the set moves under resampling — measured | used wherever a component must be read by a human [11] |
| Forward selection + nested CV (M3, aware shortlist) | optimises the practical target; nested CV gives a leak-free estimate | greedy, so a local optimum; unstable across folds; the same-data leak is this family's classic failure [7] | the standard tool when prediction, not description, is the goal [12, 13] |
| Logistic regression, Random Forest, HistGB (M4, classification) | three biases — linear, bagged, boosted — so a result across all three is not one model's artefact | trees unstable at n = 198 and need tuning; all three predict an administrative label | Random Forest is the default tabular baseline [19] |
| K-means, GMM check (M5, archetypes) | simple, fast, easy to describe to a policy audience | assumes spherical, comparable clusters; k not identified; boundary countries unstable (GMM ARI 0.58) | stability-based k selection is standard without labels [8, 9, 15, 16] |
:::

**Table&nbsp;1.** The models used here, what each is for, and where else it is used this way.

## 3. Data

### 3.1 Source and scope

All indicators come from the World Bank's World Development Indicators (WDI) [20], retrieved through the official API (`api.worldbank.org/v2`) and released under CC BY 4.0. We use an annual panel of 198 economies over 2005–2022. Economies are the World Bank's reporting entities, which include a small number of aggregates and territories (for example Curaçao, Andorra and Hong Kong SAR); this is a deliberate choice, because the policy question is about *places that appear in the data*, not about sovereign states, and excluding small high-income territories would mechanically truncate the right tail of the index. The consequences of this choice are examined in Section 6.5.

Two numbers describe the source better than the coverage figures alone. The WDI collection currently holds **1,498 series** (World Bank Data Catalog API, queried 18 November 2026) for 217 economies, back to the 1960s; the 28 used here are therefore 1.9% of what the provider publishes. That selection is not ours alone — the World Bank admits a series to WDI only after its own four-part filter of coverage, methods, user relevance and ease of use [21] — so the “full” indicator set of this paper is already an editorial product rather than a census. The same source is the standard input to the indices against which this paper is measured: the income and schooling components of the HDI, the indicators behind the MPI and the SDG monitoring series all draw on it [1, 2, 3]. Uptake is correspondingly large but diffuse: the WDI dataset record is cited about 6,800 times in OpenAlex, the OECD/JRC composite-indicator handbook about 6,700 times, and the two papers that established the PCA wealth index about 7,100 and 3,800 times (OpenAlex, 18 November 2026) [4, 17, 18].

The 28 indicators were chosen to span the six domains that development frameworks routinely cover — economy, human development, urbanisation and energy, environment, digital infrastructure, and agriculture and food — while keeping each series' cross-country coverage high enough to be usable. Table&nbsp;A.1 in Appendix A lists them with their units, transformations and domains; the set is fixed in advance of any modelling, which matters for the interpretation of the stability results in Section 5.1.

### 3.2 Preprocessing

The pipeline has four steps, applied in this order.

**Coverage filter.** Country-years with fewer than 60% of the 28 indicators observed are dropped: 389 of 3,906 candidate rows, leaving **3,517 country-years**. This removes the tail of the panel where a row would be mostly imputed, at the cost of thinning the earliest years for the poorest countries.

**Transformation.** Monetary and right-skewed series (GDP per capita, GNI per capita, health spending, CO<sub>2</sub> emissions, energy use, and similar) are passed through log(1 + x); strictly bounded series (life expectancy, enrolment ratios, electricity access, shares) are left on their natural scale. All series are then winsorised at the 1st and 99th percentiles, which clips **1,572 values** — 1.6% of the panel — so that a single reporting error in a small economy cannot dominate a component.

**Aggregation to the unit of analysis.** The unit is the **country**, not the country-year. For each economy we take the mean over 2005–2022 of each transformed indicator. Three reasons justify this choice. First, the policy object is the country; a monitoring dashboard ranks places, not place-years. Second, annual WDI values for low-income economies are sparse and revision-prone, so a country mean is a better estimate of a country's position than any single year. Third, and most importantly for our purposes, aggregating makes the analysis a *cross-sectional* problem with 198 observations, which is the regime in which stability questions can actually be answered by bootstrap: resampling years would not test what we want to test.

**Imputation and standardisation.** After averaging, 3.6% of the country × indicator cells remain missing (198 of 5,544). These are filled by k-nearest-neighbour imputation with k = 5, computed on the country profiles in the transformed space, so that a missing value is borrowed from countries with similar observed profiles rather than from the global mean. The imputed matrix is then z-scored across countries, giving the 198 × 28 matrix **Z** on which everything downstream operates.

Two properties of this pipeline deserve emphasis. It imputes **once**, at the level of the analysis, rather than imputing the panel and then averaging (which would propagate imputation error into the means); and it imputes **after** transformation, so the borrowed values live in the same space as the observed ones.

### 3.3 Missingness, and why it is a finding rather than a nuisance

Figure&nbsp;1a ranks the 28 indicators by missingness in the panel. The distribution is not uniform and it is not random: tertiary enrolment is missing for 34.5% of country-years, secondary enrolment for 27.8%, energy use per capita for 22.6%, undernourishment for 15.2%, and trade openness for 12.7%, while urbanisation, life expectancy, electricity access, agricultural land and forest cover are essentially complete. The sparse indicators are *capacity* indicators — they require a statistical apparatus, a survey programme or an energy balance — and the countries that lack them are, on average, the ones with the lowest index. Figure&nbsp;1b shows the panel's coverage by year: flat at 93–94% through 2020, then falling to 87.9% in 2022, the signature of reporting lags in the most recent vintages.

![**Figure&nbsp;1.** (a) Missingness of each indicator in the raw panel, coloured by domain. Capacity-intensive indicators — enrolment, energy balances, dietary surveys — are both the sparsest and the most concentrated in low-income countries. (b) Share of observed cells by year; the drop in 2021–2022 is a reporting lag, not a change in the world.](../results/figures/F1_data_overview.png)

This matters for the argument of the paper in a way that is easy to miss. A paper that recommends "monitor countries with 28 indicators" is recommending something that is unavailable exactly where monitoring is needed. A paper that shows 12 indicators suffice — and can say *which* 12, and how much is lost by using them — is recommending something that can actually be fielded. Section 5.2 is therefore not a compression exercise for its own sake; it is the part of the analysis that addresses the data constraint shown in Figure&nbsp;1a.

### 3.4 What the data cannot support

We flag three limits now, because they bound the claims in Section 6. **Revisions.** WDI series are revised retroactively, so the panel is a snapshot, not an archive; the retrieval date and a hash of the raw file are recorded in Appendix B. **Construct validity.** Every indicator is a proxy. Internet users measures subscriptions or survey responses, not use; enrolment measures attendance on paper, not learning. An index built on proxies inherits their biases, and the fact that it is stable does not make it valid. **Coverage of the informal economy.** GDP and employment series systematically undercount informal activity, which is a larger share of the economy in exactly the low-income countries where the index has the least data. No statistical fix exists for this inside WDI; it is a limitation to be stated, not solved.

## 4. Methods

The analysis has six stages. Each is chosen so that the quantity it produces answers one of the three research questions, and each is paired with the check that would falsify it.

### 4.1 The development index (M1)

Let **Z** be the 198 × 28 standardised country matrix. We compute the principal components of its correlation matrix, i.e. the eigen-decomposition of **Z**ᵀ**Z**/(n − 1), and define the development index as the first component score

<p align="center">index<sub>i</sub> = <b>z</b><sub>i</sub><sup>⊤</sup> <b>v</b><sub>1</sub></p>

where **v**<sub>1</sub> is the first eigenvector. Its sign is fixed by a convention rather than by the solver: we require the loading on GDP per capita to be positive, so that "higher index" always means "more developed" and the sign cannot flip between runs of the software. Because the eigenvectors are dense, the index is a weighted sum of all 28 indicators; its interpretability therefore comes from the *pattern* of loadings (Figure&nbsp;2c), not from a small support.

The variance share of the first component, λ<sub>1</sub>/28, is reported but not trusted as evidence of unidimensionality, for the reason given in Section 1.3. The evidence we do trust is the stability of the *ordering*, measured three ways in Section 5.1.

### 4.2 A label-blind shortlist (M2)

The first of the two selection criteria never looks at any outcome. We run sparse PCA [11] in the single-component form

<p align="center">min<sub><b>w</b></sub> ½ ‖<b>Z</b> − <b>Z</b><b>w</b><b>w</b><sup>⊤</sup>‖<sub>F</sub><sup>2</sup> + α‖<b>w</b>‖<sub>1</sub>  subject to ‖<b>w</b>‖<sub>2</sub> = 1</p>

sweeping the penalty α over a fixed grid from 0.005 to 1.5. Each α gives a support — the set of indicators with non-zero weight — and a fidelity loss, measured as the fraction of first-component variance retained. We choose α\* at the **knee** of that trade-off curve, i.e. the point of maximum perpendicular distance from the chord joining the two ends of the curve, which is the standard elbow rule applied to a sparsity–fidelity frontier. The resulting support is the label-blind shortlist.

Because a support is a discrete object, it needs its own stability measure: we repeat the sparse PCA on 200 bootstrap resamples of the countries and record, for each indicator, the **frequency with which it is selected**. An indicator selected in ≥ 80% of resamples is called stable. The shortlist is reported together with these frequencies, and its overlap with the label-aware shortlist is measured by the Jaccard index |A ∩ B| / |A ∪ B|.

### 4.3 A label-aware shortlist (M3)

The second criterion asks a different question: which indicators best *predict the income group*? We use greedy forward selection on balanced accuracy, with a Random Forest [22] of 120 trees as the wrapper, adding one indicator at a time for k = 1 … 14 and recording the selected order. Balanced accuracy is the scoring rule because the smallest class (low income, n = 25) would otherwise be nearly invisible to plain accuracy.

Two versions of this curve are produced. The **global** version selects on all 198 countries with 5-fold cross-validation and is the optimistic, conventional estimate. The **nested** version runs selection inside the training fold only: for each of 5 outer folds (seed 7) and each k, the forward path is rebuilt using inner 3-fold cross-validation (seed 11) on the training data alone, and the resulting k-subset is then trained on the outer training fold and scored on the held-out outer fold. Averaging over folds gives a curve in which the reported accuracy was never informed by the selection. The gap between the two curves at a given k is the selection bias for that k.

### 4.4 Classification (M4)

The shortlists are evaluated as predictors of the four World Bank income groups (high 68, upper-middle 58, lower-middle 47, low 25; classification as of the 2024 fiscal year). Four specifications are compared — all 28 indicators, the label-blind 12, the label-aware 12, and the label-aware 3 — each under three models: multinomial logistic regression on standardised inputs, a Random Forest of 400 trees [22], and histogram gradient boosting (300 iterations). A one-hidden-layer perceptron (64 ReLU units, standardised inputs, weight decay 1e-3) is added as a neural baseline; no deep-learning framework is used, because at n = 198 the parameter count a deeper network needs is not identifiable, and Section 5.3 shows the shallow network already loses to the trees. All are evaluated with `RepeatedStratifiedKFold(n_splits=5, n_repeats=5)` and we report accuracy, balanced accuracy and macro-F1 as mean ± standard deviation over the 25 folds. Repeated stratification matters at n = 198 with a class of 25: a single split can move macro-F1 by several points.

### 4.5 Archetypes (M5)

Countries are clustered in the (PC1, PC2) plane rather than in the full 28-dimensional space, for two reasons: the two components carry 56.6% of the variance after rotation and are the space in which the typology will be displayed, and clustering in 28 dimensions with n = 198 is dominated by the noisiest directions. K-means [23] is run for k = 2 … 6 with 20 initialisations each.

Three diagnostics are reported per k: silhouette [14], the adjusted Rand index [16] against the income groups (as an external sanity check, not a target), and **bootstrap stability**: 200 resamples of the countries, K-means refitted, and ARI between the original partition and the resampled one evaluated on the shared observations. The number of archetypes is chosen by bootstrap stability, subject to a silhouette that is not degenerate. When two values of k are statistically tied on stability — as k = 2 and k = 3 are here — we report the tie and break it on the content of the partition and on the external criterion, and we say so in Section 5.4. Clusters are then labelled by their mean index, which makes the naming deterministic.

Each archetype is profiled by its mean z-score on all 28 indicators. Because a *low* value is a good outcome for the indicators that load negatively (mortality, undernourishment), the profile is oriented before it is read: each indicator's z-score is multiplied by the sign of its index loading, so that higher always means more developed. The three lowest and two highest oriented values are then extracted — this is what turns a cluster label into a policy statement.

### 4.6 Dynamics (M6)

The index is recomputed twice, on 2005–2009 and on 2015–2019, using the same pipeline but restricted years, and each window index is sign-aligned to the full-window index by flipping it if its correlation with the reference is negative. Comparing ranks across the two windows gives a change of position that is not an artefact of a single year's reporting.

Three quantities are reported: the Spearman correlation between the two windows (persistence), the share of countries moving more than 10 or 20 rank places (churn), and the largest gains and losses with their index-point changes (episodes). Because the windows share no years, a country that rises has done so in the underlying data, not through a definitional overlap. The two indices are, however, separately fitted, so they have their own location and scale; index *points* are therefore comparable across countries within a window but only ordinally across windows, and rank change is the statistic we rely on.

### 4.7 Robustness (M7)

Five one-line checks are run, each answering a question a referee would ask; the penalty check is run at two alternative values. Is the typology an artefact of K-means? (Gaussian mixture with full covariance, ARI against the K-means partition.) Is it an artefact of clustering on two components? (K-means on all 28 dimensions, ARI.) Is the shortlist an artefact of the particular α? (Rerun at α = 0.25 and α = 0.50, Jaccard against the base set.) Is it driven by one region? (Drop East Asia & Pacific, which contains most of the catch-up stories, and rerun.) Does a three-component index classify better than all 28 indicators? (Random Forest macro-F1 on PC1–PC3 versus on **Z**.)

### 4.8 Software and reproducibility

Everything is implemented in Python 3.10 with NumPy, pandas, SciPy, scikit-learn [19] and Matplotlib. Bootstrap B = 200 throughout, and every resample is seeded. The pipeline is eight scripts run in order; each writes its metrics to JSON and its tables to both CSV and LaTeX, so every number in this paper is traceable to a file. Total runtime on a 16-core laptop CPU is under fifteen minutes, single-threaded, with no GPU. Appendix B gives the commands and the recorded checksums.

## 5. Results

### 5.1 One dominant axis, and an ordering that does not depend on where you look

The first principal component of the 198 × 28 correlation matrix carries 47.9% of the total variance, and the first two carry 56.6% (Table&nbsp;2). Reported alone, that number invites the objection of Section 1.3 — it is a property of the indicator set — so it is not our evidence. The evidence is that the *ordering* it induces is insensitive to three very different perturbations, and that all three are reported as correlations between rankings, not as shares of variance.

| Quantity | Value |
|:--|:--|
| Countries × indicators | 198 × 28 |
| Eigenvalues λ<sub>1</sub> … λ<sub>6</sub> | 13.48, 2.44, 1.75, 1.59, 1.38, 1.07 |
| Variance explained by PC1 | 47.90% (95% bootstrap CI 44.4–51.6%) |
| Variance explained by PC1 + PC2 | 56.57% |
| Bootstrap: Spearman(index, index of a resampled world), median / minimum | 0.9993 / 0.9974 |
| Leave one indicator out: Spearman, median / minimum | 0.9993 / 0.9986 |
| Four time windows: Spearman over the 10 pairs, median / minimum | 0.9935 / 0.9826 |
| Variance explained by PC1 inside each window (2005–09, 2010–14, 2015–19, 2015–22) | 47.6%, 47.4%, 46.0%, 46.0% |
| Spearman(index, GDP per capita) | 0.95 |

**Table&nbsp;2.** Stability of the development index. All entries are cross-country rankings compared by Spearman correlation, n = 198 unless the window is narrower.

Deleting any single indicator moves the ranking by at most ρ = 0.9986 (worst case: urbanisation), which says that no single series is load-bearing. Resampling countries moves it by at most ρ = 0.9974. Changing the time window moves it much more — down to ρ = 0.9826 between 2005–09 and 2015–22 — but that is a genuine change of the underlying data over seventeen years, not an artefact: the four windows share no years, so anything common to them is structural.

![**Figure&nbsp;2.** (a) Scree plot; the first component is large and the rest decay slowly, which is why the variance share is reported but not used as evidence of unidimensionality. (b) Index by World Bank income group; the ordering is almost monotone, but the two middle groups overlap. (c) Loadings of all 28 indicators on PC1, coloured by domain.](../results/figures/F2_index.png)

Figure&nbsp;2c is the loadings picture, and it does the interpretative work. The positive pole is a bundle of income, connectivity, schooling and infrastructure; the negative pole is under-five mortality, maternal mortality, undernourishment and — with the *opposite* sign to what a policy reading would want — renewable energy share and PM2.5 exposure. The last two are the standard failure mode of data-driven composite indices: they load with the income gradient because poor countries burn biomass and rich countries outsource their emissions, not because dirty air is a development good. We flag it here rather than hiding it, and Section 5.4 shows the consequence for the archetype profiles.

Ranking the countries produces a familiar top and bottom: the Netherlands, Luxembourg, Singapore, Belgium and the UAE at the top; Niger, the Democratic Republic of the Congo, Chad, Somalia and the Central African Republic at the bottom (Table&nbsp;A.2). The correlation with GDP per capita is ρ = 0.95 — high, but not 1: the index is not a relabelled income measure, and the 5% of rank variance it does not share with income is where the schooling and health indicators earn their place.

### 5.2 How few indicators: two criteria, two answers that barely overlap

We ran the two selection procedures of Sections 4.2 and 4.3. The label-blind sparse PCA puts its knee at α\* = 0.35, keeping **12 indicators**; the two components of that restricted model explain 38.2% of the variance against 56.3% for the dense ones — the price of sparsity, drawn as a frontier in Figure&nbsp;3a. The label-aware forward path adds indicators one at a time and reaches its own 12 at k = 12.

![**Figure&nbsp;3.** (a) Sparsity–fidelity frontier: as the constraint tightens and fewer indicators survive, the variance explained by the restricted two-component model falls; α\* = 0.35 (12 indicators) is the knee. (b) Selection frequency of all 28 indicators: blue is the label-blind bootstrap (200 resamples), orange the label-aware path across five outer folds. Bold labels are the indicators both criteria select.](../results/figures/F3_selection.png)

The two shortlists are not the same list. Their Jaccard overlap is **0.143** — they agree on two indicators, GDP per capita and internet users, and on nothing else. The label-blind set is a *welfare* set: GNI per capita, life expectancy, under-five and maternal mortality, secondary and tertiary enrolment, energy use, electricity access, renewable energy share, internet users, undernourishment. The label-aware set is a *predictive* set: GDP per capita, mobile subscriptions, GDP growth, PM2.5 exposure, trade openness, fixed broadband, health spending, energy use, measles immunisation, renewable electricity, CO<sub>2</sub> emissions, internet users. The first describes what a country is like; the second predicts which income box it will be filed in.

| | Label-blind (sparse PCA) | Label-aware (forward selection) |
|:--|:--|:--|
| Indicators retained | 12 (α\* = 0.35) | 12 (also evaluated at 3) |
| Variance explained by its two components | 38.2% (dense: 56.3%) | — |
| Spearman with the full 28-indicator index | **0.9935** | 0.9795 (k = 12), 0.9170 (k = 3) |
| Members selected in ≥ 90% of resamples | 6 of 12 | 1 of 12 |
| Balanced accuracy, selection inside the fold | — (no label used) | 0.825 (k = 12), 0.803 (k = 3) |
| Balanced accuracy, selection on the full sample | — | 0.868 (k = 12), 0.849 (k = 3) |
| Jaccard overlap with the other criterion | 0.143 | 0.143 |

**Table&nbsp;3.** The two shortlists. Fidelity is the Spearman correlation between a one-component index built from the shortlist alone and the full 28-indicator index; the variance row is the fraction of total variance explained by the restricted *two*-component model, which is the quantity the sparse solver is asked to preserve.

Three things in this table matter more than the sizes. First, **the label-blind shortlist is almost perfectly faithful**: an index computed from 12 of the 28 indicators ranks the countries at ρ = 0.9935 with the full index. It loses 0.65% of the ordering, and it costs 18 points of variance — a much bigger loss than the ranking suggests, because the discarded directions are real but do not reorder the countries.

Second, **the label-aware shortlist is far less stable, and its apparent skill is mostly selection bias**. Only one of its 12 members (GDP per capita) survives in more than 90% of resamples, against six for the label-blind set; across the five outer folds the forward path picks a different set nearly every time. The row that matters is the gap between 0.849 and 0.803 balanced accuracy at k = 3: when selection is moved inside the training fold, roughly half of the apparent skill of the three-indicator screen disappears. This is the failure mode named in Section 1.3, and it is worth stating plainly: **a three-indicator income screen looks better than it is**, and the honest number for it is about 0.80 balanced accuracy, not 0.85.

Third, the fidelity ordering and the usefulness ordering disagree. The label-aware 3-indicator set retains ρ = 0.917 of the ranking — good, but visibly worse than the 12-indicator blind set — while classifying slightly *better*. That is not a contradiction: ranking is a one-dimensional problem solved best by the directions that carry most variance, and classification is a boundary problem solved best by the indicators on which the classes differ.

### 5.3 Compressing to twelve costs almost nothing; compressing to three costs something we can measure

Table&nbsp;4 reports the classification of the four income groups under the repeated stratified cross-validation of Section 4.4.

| Specification | d | Model | Accuracy | Balanced accuracy | Macro-F1 |
|:--|--:|:--|--:|--:|--:|
| All 28 indicators | 28 | Logistic regression | 0.790 ± 0.051 | 0.769 ± 0.061 | 0.770 ± 0.059 |
| All 28 indicators | 28 | Random Forest | **0.845 ± 0.055** | **0.828 ± 0.068** | **0.834 ± 0.067** |
| All 28 indicators | 28 | Gradient boosting | 0.836 ± 0.048 | 0.813 ± 0.061 | 0.819 ± 0.061 |
| Shortlist 12 (label-blind) | 12 | Logistic regression | 0.803 ± 0.056 | 0.781 ± 0.072 | 0.778 ± 0.068 |
| Shortlist 12 (label-blind) | 12 | Random Forest | 0.840 ± 0.062 | 0.816 ± 0.076 | 0.821 ± 0.080 |
| Shortlist 12 (label-blind) | 12 | Gradient boosting | 0.825 ± 0.059 | 0.793 ± 0.071 | 0.795 ± 0.078 |
| Shortlist 12 (label-aware) | 12 | Random Forest | 0.865 ± 0.061 | 0.850 ± 0.072 | 0.857 ± 0.068 |
| Shortlist 3 (label-aware) | 3 | Random Forest | 0.880 ± 0.061 | 0.862 ± 0.071 | 0.866 ± 0.069 |
| Shortlist 3 (label-aware) | 3 | Logistic regression | 0.841 ± 0.049 | 0.813 ± 0.054 | 0.819 ± 0.056 |
| Shortlist 3 (label-aware) | 3 | Gradient boosting | 0.840 ± 0.068 | 0.820 ± 0.074 | 0.823 ± 0.077 |

**Table&nbsp;4.** Predicting the four World Bank income groups (n = 198; 68 / 58 / 47 / 25), repeated stratified 5-fold cross-validation with 5 repeats. Mean ± standard deviation over the 25 folds. The last two rows of the search order are shown for completeness; the gradient-boosting rows for the 12-indicator label-aware set are omitted for space and lie between the other two models.

The headline comparison is the honest one: **all 28 indicators reach macro-F1 0.834 and the label-blind 12 reach 0.821**, a difference of 0.013 that is well inside the fold-to-fold standard deviation of about 0.07. Dropping sixteen indicators costs nothing measurable. The label-aware rows look better — 0.857 and 0.866 — but they are contaminated by the selection of Section 5.2 and should be read as upper bounds; Table&nbsp;3 gives the corrected numbers. The neural baseline behaves exactly as the sample size predicts: under the same 25-fold protocol a 64-unit perceptron reaches macro-F1 0.763 on all 28 indicators and 0.757 on the 12-indicator shortlist — 0.05 to 0.07 below the Random Forest — and 0.853 on the three-indicator screen, where it is no better than the trees. Capacity buys nothing here: the information is in the indicators, not in the function class.

![**Figure&nbsp;4.** (a) Balanced accuracy against shortlist size under the two protocols: the grey curve selects on the full sample, the blue curve selects inside each training fold, and the vertical gap between them at a given k is the selection bias. The dashed line marks all 28 indicators under the same cross-validation as the grey curve. (b) Macro-F1 of every specification and model. (c) Confusion matrix of the 12-indicator Random Forest, summed over the 25 folds.](../results/figures/F4_classification.png)

Figure&nbsp;4a is the paper's methodological centre. The grey curve rises to a local maximum at k = 3 and then flattens at about 0.87; the blue curve is irregular, sits 0.02–0.05 below it, and peaks near k = 9–14 rather than k = 3. Two conclusions follow. The *shape* of the curve is not trustworthy below about k = 6: the apparent peak at three indicators is where selection bias is largest relative to the signal, not where the information saturates. And the honest curve still reaches 0.845 balanced accuracy at k = 9–10, which is essentially the all-28 value — the plateau is real, the location of the peak is not.

The per-class picture (Figure&nbsp;4c) explains why twelve is enough but three is not. The 12-indicator model recovers high-income countries at 0.912 recall and the low-income group at 0.728; nearly all of its error is between neighbouring classes. Its confusion matrix has 30 high-income countries predicted upper-middle, 48 lower-middle predicted upper-middle, and 29 low predicted lower-middle — a pattern of *tied* neighbours, which is what a monotone development index should produce when the class boundaries are administrative rather than developmental. The three-indicator model makes the same neighbouring errors but pushes five upper-middle countries into the low class and six lower-middle into high: it does not merely blur the boundary, it occasionally jumps the line.

### 5.4 Three archetypes, and why three rather than two or four

Clustering the country cloud in the (PC1, PC2) plane gives the diagnostics in Table&nbsp;5 for k = 2 … 6. Silhouette is highest at k = 2 (0.522) and declines slowly; bootstrap stability is essentially flat from k = 2 to k = 3 (0.952 and 0.953) and then collapses — k = 4 falls to 0.823 with a fifth percentile of 0.498, meaning that in one resample in twenty the four-way partition barely resembles the original.

| k | Silhouette | ARI vs income groups | Bootstrap ARI (mean, 5th pct) |
|--:|--:|--:|:--|
| 2 | 0.522 | 0.347 | 0.952 (0.859) |
| 3 | 0.432 | **0.565** | **0.953 (0.825)** |
| 4 | 0.424 | 0.447 | 0.823 (0.498) |
| 5 | 0.433 | 0.439 | 0.672 (0.458) |
| 6 | 0.395 | 0.335 | 0.711 (0.480) |

**Table&nbsp;5.** Choosing the number of archetypes. Stability is the primary criterion; silhouette and the agreement with the (unused) income labels are reported as secondary evidence. k = 2 and k = 3 are statistically indistinguishable on stability.

On stability alone the answer would be "two, or maybe three" — the two values differ by 0.001. We choose **k = 3** on two grounds, and the reader should know that this is a tie broken by argument rather than by the statistic. First, k = 2 partitions the world into rich and not-rich, which is a restatement of the income classification and adds no policy content; k = 3 introduces an interior type, which is where most of the world's population lives and where the targeting question is hardest. Second, k = 3 agrees much better with the external income criterion (ARI 0.565 against 0.347), which is weak evidence that the extra split is picking up structure rather than noise. Beyond k = 3 the evidence stops supporting either more types or a cleaner split.

![**Figure&nbsp;5.** (a) The 198 countries in the (index, PC2) plane, coloured by archetype. (b) Mean z-score of each archetype on all 28 indicators, ordered from left to right by how much they separate the types; note that a low value is a good outcome for the indicators that load negatively (under-five mortality, undernourishment) and that CO<sub>2</sub> emissions and renewable energy share follow the income gradient rather than a welfare reading. (c) Bootstrap stability and silhouette against k; the dotted line is the chosen k = 3.](../results/figures/F5_archetypes.png)

| Archetype | n | Mean index | High | Upper-middle | Lower-middle | Low | Gaps | Strengths |
|:--|--:|--:|--:|--:|--:|--:|:--|:--|
| Advanced | 66 | 4.01 | 61 | 5 | 0 | 0 | unemployment, forest cover, water withdrawal | fixed broadband, internet users |
| Mature middle | 73 | 0.20 | 7 | 52 | 12 | 2 | fixed broadband, CO<sub>2</sub> emissions, internet users | renewable energy share, electricity access |
| Emerging frontier | 59 | −4.72 | 0 | 1 | 35 | 23 | electricity access, secondary enrolment, life expectancy | unemployment, forest cover |

**Table&nbsp;6.** The three archetypes. *Gaps* are the three indicators on which the type sits furthest below the global mean, after orienting every indicator so that higher means more developed; *strengths* the two on which it sits furthest above. The orientation is statistical, so a "gap" in CO<sub>2</sub> emissions is not a policy deficit. Income-group counts give the composition of each type.

The profiles are the point of the table. The **advanced** type is not simply "rich": it is populated by 61 of the 68 high-income countries, and its gaps are not income-related at all — unemployment, forest cover and water withdrawal are where it underperforms the global mean, which is what a typology built on development indicators rather than on income would be expected to show. The **mature middle** is the type that the income classification cannot see: 52 upper-middle countries, an index near zero, and a distinctive gap in fixed broadband and internet users — the infrastructure that separates it from the advanced group rather than income. The **emerging frontier** has 35 lower-middle and 23 low-income countries and its gaps are the textbook ones: electricity access, secondary enrolment and life expectancy. Its "strengths" in unemployment and forest cover are the statistical orientation showing through again, and should not be read as policy achievements — measured unemployment is low in poor economies because most work is informal, which is precisely the measurement problem flagged in Section 3.4.

### 5.5 Persistence, with the exceptions that matter

Recomputing the index on 2005–09 and 2015–19 gives two rankings that agree at ρ = 0.986 (Figure&nbsp;6b). Twenty-six percent of countries move by more than ten rank places and 5.1% by more than twenty — so the ordering is persistent but not frozen, and the churn is concentrated where the data are thinnest.

![**Figure&nbsp;6.** (a) The nine largest gains and nine largest losses in rank between the two windows. (b) Index in 2015–19 against index in 2005–09; points off the diagonal are countries whose position changed, and the four labelled ones are discussed in the text. Each window index has its own scale, so the axes are comparable in ordering rather than in units.](../results/figures/F6_dynamics.png)

The largest movements are not noise. China gains 32 rank places (from 104th to 72nd), Georgia 26, Costa Rica 24, Qatar 22 and Peru 20. On the other side, Venezuela loses 34 places (76th to 110th), Aruba 24, Jamaica 23, Syria 22 (114th to 136th), Libya 21 and Yemen 19. The index was built without any information about crises, sanctions, conflict or growth episodes, and it recovers the two development collapses of the period — Venezuela's and Syria's — as the largest negative movements on the list, alongside the Chinese and Georgian catch-up as the largest positive ones. That is external validation of a kind that a bootstrap cannot provide: a noisy index does not locate a crisis.

| Country | Rank 2005–09 | Rank 2015–19 | Change | Index change |
|:--|--:|--:|--:|--:|
| China | 104 | 72 | +32 | +1.62 |
| Georgia | 105 | 79 | +26 | +1.31 |
| Costa Rica | 90 | 66 | +24 | +1.24 |
| Qatar | 29 | 7 | +22 | +0.67 |
| Peru | 116 | 96 | +20 | +1.03 |
| Venezuela | 76 | 110 | −34 | −1.39 |
| Aruba | 41 | 65 | −24 | −1.44 |
| Jamaica | 93 | 116 | −23 | −0.85 |
| Syria | 114 | 136 | −22 | −1.58 |
| Libya | 74 | 95 | −21 | −0.87 |

**Table&nbsp;7.** The five largest gains and five largest losses in rank. Index changes are within each window's own scale and are therefore comparable across countries but only ordinally across windows. n = 196 countries observed in both windows.

A caveat has to travel with the index-point column. Each window index is a separately fitted principal component, so the two have their own location and scale — the arithmetic difference between them is not a growth rate, and a country at +1.6 is not "0.16 of the index better" in any absolute sense. Rank change is the comparable statistic, which is why the section is organised around it.

At the level of the archetypes, the average index barely moves (advanced 4.06 → 3.84, mature middle 0.11 → 0.25, emerging frontier −4.69 → −4.56), and the ordering of the three types is unchanged. The type-level picture is one of stability with individual mobility: countries move between positions inside their type and occasionally across a boundary, but the three-way partition is not being reshuffled by fifteen years of development.

### 5.6 Robustness

The six rows of Table&nbsp;8 each address a question that could have overturned a claim made above.

| Check | Result | Reading |
|:--|:--|:--|
| Typology: Gaussian mixture instead of K-means | ARI 0.582 | Both extremes identical; 27 of 198 countries change side at the advanced/mature-middle boundary |
| Typology: K-means in all 28 dimensions instead of PC1–PC2 | ARI 0.938 | Same partition; the two-component display is not a simplification that changes the answer |
| Shortlist: α = 0.25 instead of 0.35 | 14 kept, Jaccard 0.86 | Stable |
| Shortlist: α = 0.50 instead of 0.35 | 11 kept, Jaccard 0.92 | Stable |
| Shortlist: East Asia & Pacific removed | 10 kept, Jaccard 0.83 | Stable; the shortlist is not a catch-up story |
| Index: PC1–PC3 instead of all 28 indicators | Macro-F1 0.761 vs 0.834 | **Worse** — the projection destroys class-relevant information |

**Table&nbsp;8.** Robustness checks. ARI is the adjusted Rand index between partitions; Jaccard is the overlap between indicator sets.

The last two rows deserve comment. Removing an entire region — the one containing most of the catch-up narratives of the period — changes two of the twelve indicators and leaves a Jaccard overlap of 0.83, so the shortlist is a description of the global income gradient rather than of East Asian growth. And the final row is the paper's most uncomfortable result: a three-component summary of the same data, which reproduces the *ranking* at ρ ≈ 0.99, classifies income groups at 0.761 macro-F1 against 0.834 for the raw 28. The projection preserves order and destroys the dimensions that separate adjacent classes. A single number cannot carry both jobs, and the typology of Section 5.4 is not a substitute for the indicators either — it is a partition of the same two components.

### 5.7 What the experiments say about the three questions

**RQ1 — one dominant axis, stable.** Yes on both counts, with the qualification that "dominant" means 48% of variance and "stable" means the *ordering* survives (ρ ≥ 0.9826 across windows, ≥ 0.9986 across indicator deletion) while the variance share does not move much either (46.0–47.9% across windows).

**RQ2 — the smallest set that reproduces the index and does the job.** Twelve, if "does the job" means reproduce the ranking and predict the income group: the label-blind 12 give ρ = 0.9935 fidelity and macro-F1 0.821 against 0.834 for all 28. Three, if "does the job" means only a coarse screen on a linear scale — but the three-indicator result is conditional on the selection, and the leakage-free estimate is 0.80 balanced accuracy rather than 0.85.

**RQ3 — how many kinds, and who moves.** Three, at the tie-break described in Section 5.4, and the types are stable at ARI 0.953 over 200 bootstraps. Movement is real but bounded: ρ = 0.986 of the ordering persists, and the exceptions line up with documented crises.

## 6. Discussion

### 6.1 The answer, stated as a rule

The experiments support a two-tier rule for a development dashboard, and a third tier that is easy to forget.

**Tier 1 — place a country: 12 indicators.** The label-blind shortlist reproduces the 28-indicator ranking at ρ = 0.9935, six of its members survive in more than 90% of bootstrap resamples, and the set classifies income groups at macro-F1 0.821 against 0.834 for the full set. It is the defensible compression: it never looks at an outcome, so its accuracy is not contaminated, and it is stable to the choice of penalty and to the removal of a whole region.

**Tier 2 — screen a country: 3 indicators.** GDP per capita, mobile subscriptions and GDP growth give ρ = 0.917 fidelity and 0.80 balanced accuracy when the selection is done inside the training fold. This is a *screening* instrument, and the difference between the two tiers is the honest content of the paper: the screen works for coarse ordering and fails for the boundary cases, particularly the lower-middle/low boundary where the recall is 0.75 and 0.73.

**Tier 3 — diagnose a country: all 28 (or more).** The single axis classifies at 0.761 macro-F1 against 0.834 for the raw indicators, and a typology is not a substitute for the indicators either. Nothing in this paper argues that the remaining sixteen indicators are useless; they are useless *for ordering*, which is the one job the index does.

The practical implication is a division of labour rather than a reduction: 3 series for a first screen, 12 for a published ranking, 28 for the diagnostic work that follows a screening decision.

### 6.2 Fidelity and usefulness are different problems, and the Jaccard index is the evidence

The two shortlists overlap in two indicators out of twenty-two (Jaccard 0.143). This is the paper's most informative negative result, and it has a simple explanation. Fidelity is a property of the *variance structure*: the directions that carry the most total variation are the ones that reproduce the ranking, and sparse PCA finds them without ever seeing a label. Usefulness is a property of the *class boundaries*: the indicators that separate upper-middle from lower-middle income are not necessarily the ones that explain most of the spread between Norway and Niger.

The composition of each list follows from that. The blind list is a welfare list — mortality, schooling, energy, nutrition — because those are the series on which countries differ most overall. The aware list is a *screening* list — GDP per capita, mobile subscriptions, GDP growth, trade openness, PM2.5 — because those are the series on which the middle-income boundary is drawn, and because they are measurable everywhere: the three members of the tier-2 screen are missing for 1.2%, 1.2% and 1.7% of country-years respectively, against 34.5% for tertiary enrolment and 27.8% for secondary enrolment, both of which are in the fidelity list. A monitoring instrument that needs the enrolment series is an instrument that is blank exactly where the screening matters. That trade-off, not the count of indicators, is the practical content of the two-criterion analysis.

### 6.3 What the archetypes add, and what they do not

The three types are a description of a cloud that is nearly one-dimensional. In the (PC1, PC2) plane of Figure&nbsp;5a the countries form a single elongated band, and the partition cuts that band into three segments; the k = 3 solution is not a set of separated islands. What the typology adds over the ranking is the *profile* — the fact that the middle type's gap is fixed broadband and internet users while the frontier's gaps are electricity, schooling and life expectancy. That is a targeting statement that a rank alone cannot make: a country ranked 90th needs different things from a country ranked 120th even when both are "middle".

What it does not add is precision. The Gaussian-mixture check (ARI 0.582) shows that the boundary between the advanced and mature-middle types is method-dependent: both extremes are identical under the two algorithms, but 27 of 198 countries change side. Any policy use of the typology should therefore be robust to the label — a country near that boundary should be treated as near the boundary, not as a member of a type.

### 6.4 What a single axis loses

It is worth being explicit about the information the index discards, because the temptation is to treat ρ = 0.99 as "nothing is lost". Two things are lost.

The first is measured in Section 5.6: the projection that preserves the ranking destroys the class-relevant dimensions enough to drop macro-F1 from 0.834 to 0.761. The index is a good *ordering* device and a poor *classification* device, and the two tasks are not the same task.

The second is the substantive content of the other principal components. The second component (8.7% of variance) contrasts, in the loadings, the energy-and-emissions block against the human-development block; a country can be high on the index and still be an outlier on that axis. The typology of Section 5.4 uses PC2 for the display but not for the partition, so this variation is visible in Figure&nbsp;5a and not used. Whether it should be used — whether "developed but carbon-intensive" is a policy-relevant type — is a question our data can pose but not answer.

### 6.5 Sensitivity to the right tail

The unit of analysis includes the World Bank's reporting entities, which means small high-income territories: Curaçao, Andorra, Aruba, New Caledonia, Hong Kong SAR. They sit at the top of the index (Aruba is 41st in 2005–09, New Caledonia is in the mature-middle type) and they are, in several cases, tax and financial centres whose GDP and energy series reflect an offshore sector rather than a resident population. Including them has three consequences worth stating: they populate the right tail of the index, they pull the advanced type's mean upward, and they contribute a handful of the rank movements of Section 5.5 (Aruba is the second-largest loser).

We did not rerun the pipeline excluding them, and that is a gap a referee would flag. What we can say is that the shortlist is not sensitive to the removal of a much larger group — dropping the whole East Asia & Pacific region leaves 10 of 12 indicators and a Jaccard of 0.83 — so the *selection* is not driven by a handful of outliers. Whether the *rankings* would change materially is untested, and the index should be read as a ranking of reporting entities rather than of populations: it is unweighted, so India and Tuvalu count the same. A population-weighted version would answer a different question.

### 6.6 Limitations and threats to validity

**The answer is conditional on the starting set.** "Twelve suffice" means twelve of *these* twenty-eight. A different starting menu — the 231 SDG indicators, or a set weighted towards governance and institutions — would produce a different twelve and possibly a different first component. The stability results bound the sensitivity to perturbation of *this* set, not to a change of set.

**The labels are administrative.** Income groups are thresholds set by the World Bank, revised over time, and not a developmental ground truth. The classification task is therefore a proxy task: a model that predicts them well is predicting an administrative construct, and the 0.83 macro-F1 should be read as "reproduces the World Bank's classification", not "identifies development".

**One data source, correlated measurement error.** All 28 series come from the same provider and, in many cases, the same national statistical offices. Errors are shared across indicators — a country that under-reports GDP likely under-reports related series — which inflates the apparent unidimensionality. The 47.9% first-component share is, in part, a property of a single measurement system.

**Linearity and causality.** PCA is a linear projection; the index is a description, not a model of how development happens. Nothing in this paper supports a claim that raising mobile subscriptions raises the index. The index-point changes of Section 5.5 are movements in a descriptive statistic.

**Imputation at the poorest end.** 3.6% of country-level cells were imputed by k-NN, and the neighbours are other countries with similar profiles. For the countries with the least data, the imputed values pull them towards their regional peers, which is precisely where an index can quietly manufacture convergence. The neighbouring-rank errors in Figure&nbsp;4c may be partly this.

**Sample size.** n = 198 limits the resolution of everything: a class of 25 countries cannot support a confident estimate of per-class recall, and the fold-to-fold standard deviations of Table&nbsp;4 (≈0.07) are the honest scale against which any difference in that table should be judged.

### 6.7 Where this question sits in practice

Composite indices are not an academic genre; they are the operating instrument of development policy. National Human Development Reports are built on the HDI, national poverty plans cite the MPI, and budget debates quote the SDG index — and each of these is a weighted summary of a longer indicator list, chosen by judgement. The question this paper asks is therefore already asked in practice, but answered by committee rather than by test: *which series must a country report before it can be placed on the map?*

Three developments make the question sharper now than a decade ago. Statistical capacity has become the binding constraint: Serajuddin et al. [24] show that dozens of countries cannot measure poverty trends at all, so an index that needs 28 series is least available precisely where targeting matters most. The provider itself now filters, and says so [21]: the “full” WDI is a curated subset, not a census. And data-driven weighting has moved out of method papers into survey operations — the wealth index computed by PCA on household assets is a standard module in large survey programmes [17, 18] — so the technique is no longer the novelty; the novelty has to be in what is asked of it.

The opposite failure mode is visible elsewhere in the same industry. Where ratings multiply without a selection discipline, the raters stop agreeing: ESG scores from major providers correlate at roughly half, largely because they choose different indicators [25]. That is the outcome the two-tier rule is designed to avoid — disagreement should be about measurement error, not about which twenty-eight series each analyst happened to prefer.

### 6.8 What we would do next

Four extensions would change the answer rather than decorate it. (i) *Widen the menu*: rerun both criteria on the 231 SDG indicators, or on a set weighted towards governance and institutions. If twelve still reproduces the ranking, the result is about development data in general and not about this menu. (ii) *Break the single-source ceiling*: repeat the index on the UN SDG database, which overlaps but is not identical, and compare the two first components. Shared measurement error is the largest threat to the 47.9% share, and a second provider is the only direct test of it. (iii) *Give the index an error bar*: a one-factor model with country-specific uncertainty, or a Bayesian PCA, would replace point ranks with credible intervals — which is what a ministry needs before reallocating a budget. (iv) *Use the screen prospectively*: take the low-income countries where the 3-indicator screen and the full index disagree, and check in the next WDI release which ordering the new data supports. That last one is a forecast, and it is falsifiable — the right next step for a descriptive result.

## 7. Conclusion

Development monitoring has an abundance problem and a scarcity problem at the same time, and this paper's contribution is to show that the first is largely an appearance. A single principal component of 28 World Development Indicators explains 47.9% of the variance of the country profile, and the ranking it induces survives the deletion of any single indicator (ρ ≥ 0.9986), the resampling of countries (ρ ≥ 0.9974) and a change of time window (ρ ≥ 0.9826). The ordering of the world is not fragile.

Twelve indicators, chosen by a procedure that never sees an outcome, reproduce that ordering at ρ = 0.9935 and predict income groups as well as the full set (macro-F1 0.821 against 0.834). Three — GDP per capita, mobile subscriptions, GDP growth — reproduce it at ρ = 0.917 and classify at about 0.80 balanced accuracy once the selection is moved inside the cross-validation fold, and the difference between the 0.85 that a naive protocol reports and the 0.80 that an honest one reports is itself a result worth reporting: the three-indicator screen is real but its apparent skill is inflated by about half.

The world's countries form three stable types — advanced, mature middle, emerging frontier — where the middle type's distinctive gap is digital infrastructure and the frontier's is electrification, schooling and survival. The types are stable under resampling (ARI 0.953) but not under a change of clustering algorithm at the advanced/middle boundary (ARI 0.582), which is the right level of confidence to have in a typology built on 198 observations.

The practical upshot is a division of labour rather than a reduction. A screening instrument needs three near-complete series; a published index needs twelve; the diagnostic work of deciding what a particular country should do still needs the rest. What a ministry does not need, and what this paper argues it should stop asking for, is all twenty-eight series before it is willing to place a country on a line.

## References

[1] United Nations, *Global indicator framework for the Sustainable Development Goals and targets of the 2030 Agenda for Sustainable Development*, A/RES/71/313, 2023.

[2] United Nations Development Programme, *Human Development Report 2023/2024*, UNDP, 2024.

[3] S. Alkire and M. E. Santos, "Acute multidimensional poverty: A new index for developing countries", OPHI Working Paper 38, University of Oxford, 2010.

[4] OECD and Joint Research Centre, *Handbook on Constructing Composite Indicators: Methodology and User Guide*, OECD Publishing, Paris, 2008.

[5] A. Saltelli, M. Ratto, S. Tarantola and F. Campolongo, "Composite indicators between analysis and advocacy", *Social Indicators Research*, vol. 81, no. 1, pp. 65–80, 2007.

[6] M. Saisana, A. Saltelli and S. Tarantola, "Uncertainty and sensitivity analysis techniques as tools for the quality assessment of composite indicators", *Journal of the Royal Statistical Society: Series A*, vol. 168, no. 2, pp. 307–323, 2005.

[7] C. Ambroise and G. J. McLachlan, "Selection bias in gene extraction on the basis of microarray gene-expression data", *Proceedings of the National Academy of Sciences*, vol. 99, no. 10, pp. 6562–6566, 2002.

[8] C. Hennig, "Cluster-wise assessment of cluster stability", *Computational Statistics & Data Analysis*, vol. 52, no. 1, pp. 258–271, 2007.

[9] U. von Luxburg, "Clustering stability: an overview", *Foundations and Trends in Machine Learning*, vol. 2, no. 3, pp. 235–274, 2010.

[10] I. T. Jolliffe, *Principal Component Analysis*, 2nd ed., Springer Series in Statistics, New York, 2002.

[11] H. Zou, T. Hastie and R. Tibshirani, "Sparse principal component analysis", *Journal of Computational and Graphical Statistics*, vol. 15, no. 2, pp. 265–286, 2006.

[12] R. Kohavi and G. H. John, "Wrappers for feature subset selection", *Artificial Intelligence*, vol. 97, no. 1–2, pp. 273–324, 1997.

[13] I. Guyon and A. Elisseeff, "An introduction to variable and feature selection", *Journal of Machine Learning Research*, vol. 3, pp. 1157–1182, 2003.

[14] P. J. Rousseeuw, "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis", *Journal of Computational and Applied Mathematics*, vol. 20, pp. 53–65, 1987.

[15] A. Ben-Hur, A. Elisseeff and I. Guyon, "A stability based method for discovering structure in clustered data", *Pacific Symposium on Biocomputing*, vol. 7, pp. 6–17, 2002.

[16] L. Hubert and P. Arabie, "Comparing partitions", *Journal of Classification*, vol. 2, no. 1, pp. 193–218, 1985.

[17] D. Filmer and L. H. Pritchett, "Estimating wealth effects without expenditure data—or tears: an application to educational enrollments in states of India", *Demography*, vol. 38, no. 1, pp. 115–132, 2001.

[18] S. Vyas and L. Kumaranayake, "Constructing socio-economic status indices: how to use principal components analysis", *Health Policy and Planning*, vol. 21, no. 6, pp. 459–468, 2006.

[19] F. Pedregosa et al., "Scikit-learn: machine learning in Python", *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[20] World Bank, *World Development Indicators*, Washington, DC, 2024. Data retrieved from the WDI API, `api.worldbank.org/v2`, CC BY 4.0.

[21] World Bank, "Inside the World Development Indicators: how do we select the indicators?", Data Blog, 19 March 2025. <https://blogs.worldbank.org/en/opendata/inside-the-world-development-indicators--how-do-we-select-the-in>

[22] L. Breiman, "Random forests", *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[23] J. MacQueen, "Some methods for classification and analysis of multivariate observations", *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, pp. 281–297, 1967.

[24] R. Serajuddin, H. Uematsu, C. Wieser, N. Yoshida and A. Dabalen, "Data deprivation: another deprivation to end", Policy Research Working Paper 7252, World Bank, 2015.

[25] F. Berg, J. F. Kölbel and R. Rigobon, "Aggregate confusion: the divergence of ESG ratings", *Review of Finance*, vol. 26, no. 6, pp. 1315–1344, 2022.

## Appendix A. The indicator set and the index

::: {.twide}
| # | Indicator | Domain | Unit | log | Missing (%) | Blind 12 | Aware 12 |
|--:|:--|:--|:--|:--:|--:|:--:|:--:|
| 1 | Cereal yield | Agri-food | kg/ha | yes | 8.7 | | |
| 2 | Undernourishment | Agri-food | % of population | no | 15.1 | Y | |
| 3 | Fixed broadband | Digital | per 100 people | yes | 6.5 | | Y |
| 4 | Internet users | Digital | % of population | no | 3.7 | Y | Y |
| 5 | Mobile subscriptions | Digital | per 100 people | yes | 1.2 | | Y |
| 6 | FDI inflow | Economy | % of GDP | no | 3.8 | | |
| 7 | GDP growth | Economy | % per year | no | 1.7 | | Y |
| 8 | GDP per capita | Economy | current US$ | yes | 1.2 | Y | Y |
| 9 | GNI per capita PPP | Economy | international $ | yes | 4.0 | Y | |
| 10 | Trade openness | Economy | % of GDP | no | 12.8 | | Y |
| 11 | Unemployment | Economy | % of labour force | no | 6.9 | | |
| 12 | Agricultural land | Environment | % of land | no | 0.3 | | |
| 13 | CO<sub>2</sub> emissions per capita | Environment | t CO<sub>2</sub>e/capita | yes | 2.6 | | Y |
| 14 | Forest cover | Environment | % of land | no | 0.8 | | |
| 15 | PM2.5 exposure | Environment | µg/m³ | yes | 1.6 | | Y |
| 16 | Water withdrawal | Environment | % of resources | yes | 10.9 | | |
| 17 | Health spending | Human dev. | % of GDP | no | 3.8 | | Y |
| 18 | Life expectancy | Human dev. | years | no | 0.0 | Y | |
| 19 | Maternal mortality | Human dev. | per 100,000 | yes | 1.7 | Y | |
| 20 | Measles immunisation | Human dev. | % of 12–23m cohort | no | 2.3 | | Y |
| 21 | Secondary enrolment | Human dev. | % gross | no | 27.8 | Y | |
| 22 | Tertiary enrolment | Human dev. | % gross | no | 34.5 | Y | |
| 23 | Under-5 mortality | Human dev. | per 1,000 births | yes | 2.3 | Y | |
| 24 | Electricity access | Energy | % of population | no | 0.2 | Y | |
| 25 | Energy use per capita | Energy | kg oil equivalent | yes | 22.6 | Y | Y |
| 26 | Renewable electricity | Energy | % of electricity | no | 8.2 | | Y |
| 27 | Renewable energy share | Energy | % of final energy | no | 4.2 | Y | |
| 28 | Urbanisation | Energy | % of population | no | 0.0 | | |
:::

**Table&nbsp;A.1.** The 28 indicators, with the World Bank code and retrieval in `src/common.py`. "log" marks indicators entering the pipeline as log(1 + x); every indicator is then winsorised at the 1st and 99th percentiles and standardised. Missing (%) is the share of the 3,517 country-years with no observation, i.e. before imputation. The last two columns mark membership of the two 12-indicator shortlists of Section 5.2.

| Rank | Country | Index |
|--:|:--|--:|
| 1 | Netherlands | 5.92 |
| 2 | Luxembourg | 5.75 |
| 3 | Singapore | 5.68 |
| 4 | Belgium | 5.68 |
| 5 | United Arab Emirates | 5.68 |
| 194 | Niger | −6.89 |
| 195 | Congo, Dem. Rep. | −7.03 |
| 196 | Chad | −7.38 |
| 197 | Somalia | −7.56 |
| 198 | Central African Republic | −7.68 |

**Table&nbsp;A.2.** The top and bottom of the index, 2005–2022 country means. The full ranking is written to `results/tables/T1_index_ranking.csv`.

## Appendix B. Reproducibility

The study is eight scripts in `src/`, run in order, each writing its metrics to `results/metrics/*.json`, its tables to `results/tables/*.{csv,tex}` and its figures to `results/figures/*.{pdf,png}`. Every number in this report is regenerated by them; no number was transcribed by hand. (A ninth script, `10_slide_figs.py`, renders the two simplified presentation figures and is not part of the analysis.)

| Stage | Script | Output | Runtime |
|:--|:--|:--|--:|
| Panel construction | `01_preprocess.py` | 3,517 × 28 panel, coverage report | 4 s |
| Index and stability | `02_index.py` | E1, Table&nbsp;2, Figure&nbsp;2 | 2 s |
| The two shortlists | `03_selection.py`, `03b_fig_selection.py` | E2, Table&nbsp;3, Figure&nbsp;3, fidelity | 633 s |
| Classification | `04_classify.py` | E3, Table&nbsp;4, Figure&nbsp;4 | 95 s |
| Archetypes | `05_cluster.py` | E4, Tables 4–5, Figure&nbsp;5 | 34 s |
| Dynamics | `06_dynamics.py` | E5, Table&nbsp;7, Figure&nbsp;6 | 6 s |
| Robustness | `07_robustness.py` | E6, Table&nbsp;8 | 48 s |

Environment: Python 3.10, NumPy 2.2, pandas 2.3, SciPy 1.15, scikit-learn 1.7, Matplotlib 3.10; single-threaded, no GPU, 16-core laptop CPU. All resampling is seeded (`random_state = 0` for models, 42 for the repeated CV, 7/11 for the nested protocol); the bootstrap uses B = 200 throughout. The raw WDI pull, its retrieval date and a SHA-256 are recorded in `data/PROVENANCE.md`; the frozen response in `data/raw/wdi_panel_long.csv` lets the pipeline rerun offline.

::: {.newpage}
## Statement of Individual Contribution
:::

The three authors contributed equally. The work ran in four streams, and every author took part in all four: (i) problem framing and literature, (ii) data construction and code, (iii) experiments and robustness checks, (iv) writing, figures and review. The table records the primary owner of each stream; the percentage is the share of total effort.

::: {.tcontrib}
| Member | Student ID | Primary responsibility | Contribution |
|:--|:--|:--|--:|
| Chen Wanchuan | 26108204G | Problem framing; index construction and stability analysis; Sections 1–2 and 6 | 33.3% |
| Lu Yuqing | — | Data pipeline, imputation and coverage analysis; the two shortlist procedures; Sections 3–4 | 33.3% |
| Yuen Lok Yan | — | Classification, archetypes and dynamics; robustness checks; Sections 5 and 7; figure and table production | 33.3% |
:::

We confirm that all three authors reviewed the complete analysis, re-ran the pipeline end to end on the released code, and checked every reported number against the artefacts in `results/`. Drafting of code and text was assisted by an AI coding tool; all outputs were verified against the released artefacts by the authors, who take full responsibility for the content.

**Declaration.** The data are public (World Bank WDI, CC BY 4.0) and were retrieved through the official API. No proprietary or personal data were used. The code, the intermediate artefacts and the exact commands to regenerate every table and figure accompany this submission.

# Reducing Development Indicators While Preserving Country Rankings and Income Classification

Group 16: Chen Wanchuan, Lu Yuqing, Yuen Lok Yan

## Abstract

This study asks whether a smaller set of development indicators can preserve country rankings, retain income-group classification performance and reveal useful differences between country profiles. We analyse 28 World Development Indicators for 198 economies, using country averages over 2005 to 2022 from the official archive downloaded on 19 November 2026. Principal component analysis provides a descriptive reference index, sparse component estimation selects a shorter indicator set, and training-fold evaluation compares classification procedures. The first component explains 50.1% of standardised variance. A nine-indicator subset achieves a Spearman rank correlation of 0.992 with the full index. On the main split, macro-F1 was 0.856 to three decimal places for both the fixed nine-indicator list and the procedure that reselected 17 to 18 indicators within training folds. The full random forest scored 0.844 and the nested three-indicator screen scored 0.820. The fixed list was chosen using predictors from all economies before splitting and averaged 0.012 lower macro-F1 than within-fold reselection across ten split seeds. The results support treating the nine indicators as a candidate dashboard for this menu. Their external validity and policy usefulness still need further evaluation.

## 1 Introduction and research questions

Development monitoring combines measures of income, health, education, infrastructure and environmental conditions. These variables describe different aspects of an economy, but some also move together. Collecting a longer list can therefore add detail without changing an economy's broad position. Conversely, a short list may remove information needed to distinguish economies with similar incomes. Guidance on composite indicators stresses that selection, normalisation, weighting and sensitivity analysis must be justified together (OECD et al., 2008; Nardo et al., 2005). A stable numerical index is not automatically a valid measure of development. Index construction involves choices that are not purely statistical. The menu, the weights and the aggregation rule all embed judgements, and a composite can move from description towards advocacy (Ravallion, 2012; Saltelli, 2007). Sensitivity analysis is particularly important because small changes in indicator choice or weighting can alter rankings (Saisana et al., 2005).

Correlated development indicators may share a common direction that can be summarised with a single score. We use dimension reduction to identify that direction and regularisation to test whether a shorter list can preserve the resulting rankings. Classification tests whether the retained indicators distinguish income groups, and clustering identifies economies with similar profiles across the first two principal components.

The overarching question is how far the indicator set can be reduced without losing the information required for a specified task. We address three linked questions. First, does the selected development data contain a dominant and stable ranking direction? Second, can a shorter indicator set preserve that ranking and retain income-group classification performance when selection is evaluated on held-out economies? Third, do multivariate groupings and changes in rank provide information that a single ordering conceals?

The study's contribution is an integrated comparison of these three tasks on one documented dataset. We define success separately for compression, classification and country profiles. A high correlation with an internally constructed index measures fidelity to that index. It does not establish that the index is an external ground truth, that the selected subset is globally minimal, or that a particular policy intervention would improve development.

## 2 Data and preparation

The source is the World Bank's World Development Indicators bulk CSV archive supplied for this revision (World Bank, 2026). Its files are dated 16 November 2026. The download date is 19 November 2026. We examine 28 indicators spanning the economy, human development, urbanisation and energy, environment, digital connectivity, and agriculture and food. Appendix A lists the retained variables in readable form. These are a chosen subset of the database, not an exhaustive definition of development. The archive publishes roughly 1,500 series (World Bank, 2026), so the menu used here covers about 2% of what is available. The menu balances coverage of income, health, education, infrastructure and environment, keeps series with reasonable cross-economy reporting coverage over 2005 to 2022, and avoids near-duplicate indicators within the same dimension.

Regional aggregates are excluded using the archive's economy metadata, while economies and territories with a non-empty region remain eligible. Restricting the annual observations to 2005 to 2022 gives 3,906 candidate economy-years. We exclude observations with fewer than 17 of the 28 indicators recorded (the integer equivalent of the 60% requirement), leaving 3,517 economy-years. We then require at least 60% indicator coverage in each economy's aggregated profile, giving 198 economies. These filters limit dependence on imputation but can reduce representation of economies with weaker reporting systems (Serajuddin et al., 2015). A territory is an economy-level reporting unit here, not necessarily an independent country.

The preparation order follows the intended unit of analysis. We first apply log(1 + x) to the positive magnitude series identified in Appendix A, reducing the influence of long right tails. We then average each transformed indicator over the available years for each economy. Averaging reduces year-specific fluctuations and avoids treating repeated observations of the same economy as independent training examples. It also changes the question. The main analysis concerns average profiles over 2005 to 2022, rather than a country's current situation or an annual forecast.

At this stage, 198 of the 5,544 economy-indicator cells are missing (3.6%). For the descriptive analysis, each variable is clipped at its first and ninety-ninth percentiles across economy means, scaled using its observed mean and standard deviation, and filled using five-nearest-neighbour imputation (Troyanskaya et al., 2001). Nearest-neighbour filling preserves local structure across correlated indicators, which simple mean substitution does not, and works well at the level of missingness observed here. Each missing entry is filled by the unweighted mean from up to five neighbouring economies with that indicator observed. The distance uses indicators observed for both economies and adjusts for the number of shared observations (scikit-learn developers, n.d.-b). We standardise the indicators before computing distances so that variables measured on larger scales do not dominate the choice of neighbours. A final standardisation gives each completed variable unit variance. These choices reduce scale effects. They do not eliminate uncertainty from missing or potentially non-random observations.

![Figure 1](../results/figures/missingness.png)

Figure 1. Missingness for the 14 indicators with the highest pre-filter missingness, before and after the economy-year coverage filter, using the official 2005 to 2022 panel. Percentages concern annual observations before imputation, not the aggregated country matrix.

Figure 1 shows why preparation is necessary before modelling. Reporting completeness differs across indicators, so excluding the least complete economy-years reduces missingness but does not remove it. The graph describes the retained sample's coverage. It cannot establish why observations are missing or show that the missingness mechanism is random. The later imputation step therefore remains a modelling assumption.

The classification analysis relearns clipping thresholds, scaling and imputation using training economies only. Held-out economies are transformed using those fitted operations. For a selected subset, the classifier's preprocessing is fitted again using only the retained inputs, so its held-out predictions do not borrow information from discarded variables. Even a transformation estimated without labels can bias cross-validation if it uses held-out predictors (Moscovich & Rosset, 2022). The fixed nine-indicator comparison uses a list defined from the full predictor matrix, as described in Section 3.

## 3 Methods and their purposes

![Figure 2](../results/figures/workflow.png)

Figure 2. Analysis workflow linking preparation to ranking, classification and profile comparison. The classification branch depicts training-fold selection. The separate fixed-list benchmark takes its indicator list from the ranking branch.

Figure 2 links each research question to its analysis. The classification path shows within-fold feature selection. In an additional comparison, the full-sample ranking path supplies a fixed nine-indicator list and the classifier is trained within each fold.

To answer the first question, we fit principal component analysis to the completed standardised matrix. The first direction maximises the variance of a linear projection subject to a unit-length loading vector. Equivalently, it is the leading eigenvector of the sample covariance matrix (Jolliffe & Cadima, 2016). The index for an economy is its standardised indicator vector projected onto that direction. We orient the sign so that the loading on gross domestic product per capita is positive. The method discovers covariance structure, rather than choosing normative welfare weights (Jolliffe & Cadima, 2016).

We report the first component's variance share and assess whether the ordering is sensitive to indicator removal and resampling economies. Each leave-one-indicator-out analysis refits the component and compares its ranking with the full index using Spearman correlation. Two hundred bootstrap resamples refit the component on the completed standardised profiles. Their variance shares give a percentile interval, and their loading vectors are applied to the reference profiles to assess ranking stability. Because this bootstrap holds the initial preprocessing fixed, its uncertainty is conditional on the prepared matrix. It does not propagate every source of measurement or imputation uncertainty.

PCA extracts new features as linear combinations of the original indicators. Each component's share of total variance is its eigenvalue divided by the sum of all eigenvalues (Jolliffe & Cadima, 2016).

Sparse PCA promotes zero loadings (Zou et al., 2006; d'Aspremont et al., 2007). This lets us use its fit to select original indicator columns. For the centred, standardised matrix \(Z\), scikit-learn's sparse matrix factorisation solves the rank-one reconstruction problem below. The vector \(u\) contains reconstruction coefficients across economies and \(w\) contains indicator loadings. The \(\ell_1\) penalty on \(w\) encourages exact zeros as \(\alpha\) increases (Mairal et al., 2010; scikit-learn developers, n.d.-b).

$$
\min_{u,w\,:\, \parallel u \parallel_{2} \leq 1}\ \frac{1}{2} \parallel Z - uw^{\top} \parallel_{F}^{2} + \alpha \parallel w \parallel_{1}
$$

We retain loadings with absolute value above \(10^{-8}\), which separates exact zeros produced by the \(\ell_1\) penalty from non-zero coefficients stored at machine precision, and evaluate 13 values of \(\alpha\) from 0.05 to 12. For each non-empty support we refit ordinary PC1 using only the retained indicators. The absolute Spearman coefficient is the absolute Pearson correlation of the two rank vectors, so it measures ordering fidelity independently of the arbitrary component sign (Spearman, 1904).

Our declared selection rule chooses the smallest support on this penalty grid whose ranking correlation reaches 0.99. Ties favour the larger penalty. If a fit yields no candidate at 0.99, the procedure uses the candidate with the highest fidelity. The threshold is a practical fidelity criterion, not a significance test. This selects a candidate from the fitted regularisation path, not a proven minimum over all possible variable subsets. Two hundred bootstrap fits at the selected full-sample penalty describe how often individual variables survive. The reduced index and its fidelity are descriptive, in-sample summaries. The separate classification evaluation addresses held-out performance.

Income-group classification tests a different property. The target is the Income Group field accompanying the downloaded economy metadata, rather than an assumed historical fiscal-year classification. The World Bank bases income categories on gross national income per capita converted using the Atlas method (World Bank, n.d.). Income-related inputs therefore make this a related administrative classification task, not an independent validation of overall development. We also remove both income-per-capita indicators in a sensitivity analysis to assess reliance on that relationship.

We compare random forests with logistic regression. A random forest combines multiple classification trees (Breiman, 2001). Each tree uses a bootstrap sample of the training economies and considers a random subset of indicators at each split. The split criterion uses Gini impurity, defined at a node as one minus the sum of the squared class proportions (scikit-learn developers, n.d.-a). The fitted forest averages the trees' estimated class probabilities (scikit-learn developers, n.d.-b). Logistic regression provides a linear decision-boundary comparison. The specifications are fixed at 200 trees for the outer fits and default logistic regularisation with a maximum of 3,000 iterations. The shared five-fold stratified split approximately preserves class proportions and holds each economy out once. In the within-fold sparse procedure, preprocessing and support selection are fitted on each training partition.

The fixed nine-indicator benchmark uses the same outer folds, training-fold preprocessing and model settings as the within-fold sparse procedure.

Its list was selected once using all economies' predictors without income labels. As a result, held-out predictor values can influence which columns enter the benchmark. The reported scores therefore assess the fitted models conditional on this list.

A fully out-of-sample assessment of the list-selection workflow would require either fresh economies or repeating selection inside each training fold (Moscovich & Rosset, 2022).

A three-indicator supervised screen provides an additional comparison under a fixed input budget. Within each outer training set, greedy forward selection adds three variables using balanced accuracy in three inner stratified folds and a 60-tree forest. Every inner fold relearns preprocessing. A 200-tree forest is then fitted to the chosen variables in the outer training set and evaluated on the outer held-out economies. This nested cross-validation design selects indicators in the inner folds and evaluates the resulting selection procedure on the outer held-out folds (Varma & Simon, 2006). It estimates the performance of selecting three variables, not the performance of a single universally fixed list.

We compute each metric from the pooled held-out predictions. For each income class \(c\), precision \(P\) is the share of predictions for that class that are correct, and recall \(R\) is the share of economies in that class correctly identified. Accuracy is the overall proportion correctly classified. Balanced accuracy averages class recall, and macro-F1 averages the class-level harmonic means of precision and recall (Opitz, 2024). Here \(C = 4\). Equal weighting gives the 25 low-income economies' class the same influence on these macro summaries as the 68 high-income economies' class.

$$
\text{Balanced accuracy} = \frac{1}{C}\sum_{c=1}^{C} R_c, \quad \text{Macro F1} = \frac{1}{C}\sum_{c=1}^{C} \frac{2 P_c R_c}{P_c + R_c}
$$

For the third question, we apply K-means to the first two principal component scores. The K-means objective is the sum of squared distances from economies to their assigned cluster centroids (Lloyd, 1982). We compare two to six groups using the mean silhouette. For each economy, the silhouette is \((b - a)/\max(a, b)\), where \(a\) is its mean distance to its own cluster and \(b\) is the smallest mean distance to any other cluster (Rousseeuw, 1987). Two hundred bootstrap refits provide a stability diagnostic using the adjusted Rand index, which measures pairwise agreement between partitions after correction for chance (Hubert & Arabie, 1985). Each bootstrap model assigns all reference economies before comparison with the original partition. The component projection is held fixed, so this diagnostic concerns partition stability in that display, not the stability of the entire analytical pipeline.

Finally, we rebuild the index separately for 2005 to 2009 and 2015 to 2019, applying the same coverage and preparation rules within each window. We compare ranks only among economies available in both windows and recompute the ranks on that common population. This distinguishes changing relative position from a changing comparison set. The two indices have separately estimated directions and scales, so their numerical score differences are not treated as a common-unit measure of development progress.

## 4 Results

For the first research question, the first component explains 50.1% of total standardised variation and the first two explain 58.7%. The first direction is therefore prominent, but substantial variation remains outside it. Its bootstrap percentile interval is 46.6% to 53.7%. The minimum ranking correlation after deleting one indicator is 0.999. Correlations from the conditional bootstrap range from 0.998 to 1.000. These checks show that the ranking is stable within this indicator menu. They do not show that development is one-dimensional.

![Figure 3](../results/figures/structure.png)

Figure 3. Variance explained by the leading components and the largest positive and negative first-component loadings. Signs indicate association with the fitted index, not whether an indicator is desirable.

For the second question, the selected penalty is 12, retaining nine indicators and producing a ranking correlation of 0.992 with the full index. The selected indicators are gross domestic product per capita, gross national income per capita at purchasing power parity, life expectancy, under-5 mortality, maternal mortality, secondary enrolment, energy use per capita, internet users, and fixed broadband. This is a candidate compression relative to the chosen 28-variable reference. A high correlation does not rule out meaningful rank changes between neighbouring economies. It also says nothing about information that was never in the original menu. Note that the selected penalty sits at the upper boundary of the evaluated grid.

![Figure 4](../results/figures/selection.png)

Figure 4. Number of retained indicators and ranking fidelity along the sparse-estimation penalty path. The dashed line marks the declared 0.99 criterion. The marked candidate is selected without income labels.

| **Procedure**                                     | **Accuracy** | **Balanced accuracy** | **Macro F1** |
|---------------------------------------------------|--------------|-----------------------|--------------|
| All 28 indicators, random forest                  | 0.859        | 0.833                 | 0.844        |
| Fixed nine-indicator list, random forest          | 0.864        | 0.844                 | 0.856        |
| Within-fold sparse selection, random forest       | 0.864        | 0.849                 | 0.856        |
| Nested three-variable selection, random forest    | 0.843        | 0.813                 | 0.820        |
| Random forest without income variables            | 0.828        | 0.811                 | 0.823        |
| All 28 indicators, logistic regression            | 0.793        | 0.771                 | 0.773        |
| Fixed nine-indicator list, logistic regression    | 0.823        | 0.800                 | 0.809        |
| Within-fold sparse selection, logistic regression | 0.778        | 0.754                 | 0.758        |

Table 1. Pooled held-out predictions for 198 economies on the same five outer folds. Preparation is refitted inside each training fold for every row. The fixed rows use the nine indicators selected once on the full sample and keep the same columns in every fold. The within-fold rows reselect a sparse support inside each training fold, where the supports contain 17 to 18 indicators.

On the main split, the fixed nine-indicator and within-fold sparse forests both reach 0.856 macro-F1 to three decimal places, against 0.844 for the full forest. Both shortlists use the same split and forest size, with the selection difference described in Section 3. The nested three-variable screen reaches 0.820 under its own feature-selection rule. These scores show how the tested procedures performed at the evaluated input sizes. For logistic regression, the fixed list scores 0.809, compared with 0.773 for the full menu and 0.758 for within-fold selection. The fixed list helps logistic regression more than it helps the random forest. Sparse selection optimised for a flexible learner removes indicators that a linear boundary still needs, and a shorter list can therefore help a linear classifier where it leaves a flexible one unchanged.

![Figure 5](../results/figures/confusion.png)

Figure 5. Confusion matrices for held-out random-forest predictions using all indicators and training-fold sparse selection. Each cell shows the number of economies and its percentage within the recorded income class. Each economy contributes one prediction to each matrix. The reduced supports contain 17 to 18 indicators.

In Figure 5, rows represent recorded income classes and the diagonal entries divided by each row total are class recalls (Opitz, 2024). All errors in these two matrices occur between adjacent income categories. Both forests correctly classify 62 of 68 high-income economies and 37 of 47 lower-middle-income economies. The within-fold sparse forest identifies 20 of 25 low-income economies, compared with 18 for the full forest, while its correct upper-middle-income classifications fall from 53 to 52 out of 58. These differences show that similar overall scores can still hide different patterns of mistakes at the class level.

The nested three-variable procedure gives balanced accuracy 0.813 and macro-averaged F1 0.820. This result concerns a deliberately severe input restriction, not a discovered optimal input count. Removing both income variables gives random-forest macro-averaged F1 0.823. The comparison shows how sensitive the classification is to income-related inputs. It cannot turn an administrative income label into a comprehensive measure of social or environmental development.

| **Groups** | **Silhouette** | **Bootstrap adjusted Rand index** |
|------------|----------------|-----------------------------------|
| 2          | 0.533          | 0.938                             |
| 3          | 0.434          | 0.948                             |
| 4          | 0.433          | 0.831                             |
| 5          | 0.379          | 0.683                             |
| 6          | 0.370          | 0.593                             |

Table 2. Candidate K-means partitions in the two-component display. Stability values are means across 200 resampled fits with the projection fixed.

For the third question, the silhouette criterion selects 2 groups, containing 77 and 121 economies. The three-group partition has slightly higher bootstrap agreement (adjusted Rand index of 0.948 versus 0.938), while the two-group partition has the higher silhouette (0.533 versus 0.434). The two criteria measure different properties of the fitted partition (Hubert & Arabie, 1985; Rousseeuw, 1987). The grouping describes regions of the projected data. It does not show that countries pass through discrete development stages. In particular, the display contains only 58.7% of the variation, so apparent separation can conceal differences on omitted directions. Group profiles can guide which original indicators to inspect. They do not establish policy prescriptions.

![Figure 6](../results/figures/groups.png)

Figure 6. Economies in the first two component directions, coloured by the silhouette-selected K-means partition. The first component is oriented towards higher income. The second component has an arbitrary sign.

The first group combines below-average income, connectivity and life expectancy with above-average child and maternal mortality. The second shows the opposite broad pattern. These differences explain why the partition largely follows the leading development gradient. They support using the original indicator profiles to describe contrasting conditions, but the labels remain descriptive and should not be read as fixed stages that all economies must pass through.

Across the two non-overlapping windows, 196 economies are shared and their ranking correlation is 0.986. Two economies from the main sample lack sufficient indicator coverage in one or both windows and drop out of this comparison. On the common 196-economy set, the largest positive changes are China (+28 places), Georgia (+25 places) and Qatar (+22 places). The largest negative changes are Jamaica (−24 places), Libya (−24 places) and Venezuela (−40 places). A positive number means a higher relative position in the later window. These are descriptive changes in rank, not causal estimates or proof of absolute improvement.

![Figure 7](../results/figures/rank_changes.png)

Figure 7. Rank changes for the three largest gains and three largest losses among the 196 economies shared by both time windows. Blue markers show 2005 to 2009 and orange markers show 2015 to 2019. Signed labels give earlier rank minus later rank, so a positive value indicates a higher relative position in the later window. These deliberately selected extremes do not represent the typical economy.

Figure 7 complements the high cross-window Spearman correlation with individual changes. The coefficient summarises agreement across all shared ranks, so it can remain high when individual economies move substantially (Spearman, 1904). Both markers use ranks recomputed on the same 196 economies, so changes do not arise merely from adding or removing comparison economies. The figure addresses the change component of the third research question, while leaving the causes of those movements untested.

## 5 Discussion and limitations

The fixed nine-indicator list came from an unsupervised ranking criterion and matched the within-fold sparse procedure to three decimal places on the main split. Its use of full-sample predictors and the variation over repeated splits qualify that comparison (Moscovich & Rosset, 2022).

The practical implication is to treat the selected subset as a candidate reporting layer above the full data. A shorter display can make a broad ordering easier to communicate, while the remaining variables support examination of health, education, environmental conditions and other dimensions. The study has not measured collection costs, field usability or the consequences of replacing a real monitoring system. Therefore it does not justify instructing agencies to stop collecting the discarded variables.

Several limits constrain the conclusion. The 28-variable menu is selected by judgement and excludes governance, inequality and other important dimensions. Its variables are not independent measures. Some reflect overlapping economic processes and shared statistical systems. A dominant direction can partly represent this overlap. In addition, the index is oriented towards income, and a loading's sign is an empirical association rather than a welfare judgement. For example, an unfavourable sign on renewable energy share would not show that renewable energy reduces development. Sensitivity checks of the kind recommended for composite indicators (Saisana et al., 2005) would be a natural next step.

Country averages combine observations from unequal numbers of years. Missingness may be concentrated among economies with weaker statistical capacity, and nearest-neighbour filling can smooth genuine differences.

Log transformations and clipping also alter the geometry used by the methods. The conditional bootstrap does not capture those choices. In addition, the panel's common reporting source can produce correlated measurement error.

Results should be read as properties of this specific data preparation. They are not an uncertainty-free global ranking.

The selected penalty sits at the upper boundary of the evaluated grid, so a shorter support may exist beyond the search range. The nine-indicator candidate is the smallest support reached within the declared path, not a proven minimum across all penalty values.

Across ten seeds for stratified five-fold cross-validation, the fixed nine-indicator forest averaged 0.842 macro-F1, the within-fold sparse forest 0.855, and the full forest 0.851. Using the unrounded scores, the fixed-minus-reselected difference averaged −0.012 and ranged from −0.039 to 0.000 after rounding. The reselected supports contained 16 to 18 indicators across these runs. Repeating splits assesses sensitivity to how the same 198 economies are partitioned. An independent sample would be needed to assess the fixed list after it has been selected. The income labels describe the archive snapshot and the predictors average 2005 to 2022, so the task concerns association with that snapshot. The three-variable restriction and the ranking fidelity threshold define the comparisons evaluated here.

Further work should first repeat the complete procedure across alternative imputation and scaling rules, since the split-seed check above covers only the partitioning. It should then test the candidate subset on another period, measure the consequences of excluding income-related variables from index construction, and assess group stability while refitting the projection. These checks would address generalisation and construct validity before adding more complex predictive models. The current study demonstrates the workflow and its trade-offs, not a validated replacement for established development measures.

## 6 Conclusion

The first research question receives a qualified positive answer. A prominent direction explains 50.1% of the selected data's variation and produces a ranking that is stable under the reported conditional checks. The remaining variation prevents us from equating that direction with the whole of development.

For the second question, nine indicators on the evaluated regularisation path reproduce the full-sample ranking at correlation 0.992. On the reported split the frozen nine-indicator list achieves macro-averaged F1 0.856, the same score as a procedure that reselects its support inside every training fold, and above the 0.844 of the full random forest. The three-indicator screen falls to 0.820. These comparisons support investigating the nine indicators as a candidate dashboard. The fixed list was selected from the full predictor matrix and averaged 0.012 lower macro-averaged F1 than within-fold reselection across ten split seeds, so its performance on an independent sample remains to be established.

For the third question, the data support a 2-group descriptive partition in the two-component display and a cross-window ranking correlation of 0.986. Group profiles and relative movements complement the ordering, while remaining sensitive to representation and preparation. Taken together, the findings show why indicator reduction should be evaluated through explicit research questions. Fewer inputs may preserve a broad ranking, but neither ranking fidelity nor income classification alone establishes complete or policy-ready measurement of development. The number of indicators an office reports is ultimately a governance choice about what to monitor, and that choice is better defended by explicit fidelity and prediction criteria than by the appeal of a single number (Ravallion, 2012; Saltelli, 2007).

## References

Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5–32. https://doi.org/10.1023/A:1010933404324

d'Aspremont, A., El Ghaoui, L., Jordan, M. I., & Lanckriet, G. R. G. (2007). A direct formulation for sparse PCA using semidefinite programming. *SIAM Review, 49*(3), 434–448. https://doi.org/10.1137/050645506

Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification, 2*, 193–218. https://doi.org/10.1007/BF01908075

Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: A review and recent developments. *Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 374*(2065), Article 20150202. https://doi.org/10.1098/rsta.2015.0202

Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory, 28*(2), 129–137. https://doi.org/10.1109/TIT.1982.1056489

Mairal, J., Bach, F., Ponce, J., & Sapiro, G. (2010). Online learning for matrix factorization and sparse coding. *Journal of Machine Learning Research, 11*, 19–60. https://jmlr.org/papers/v11/mairal10a.html

Moscovich, A., & Rosset, S. (2022). On the cross-validation bias due to unsupervised preprocessing. *Journal of the Royal Statistical Society Series B: Statistical Methodology, 84*(4), 1474–1502. https://doi.org/10.1111/rssb.12537

Nardo, M., Saisana, M., Saltelli, A., Tarantola, S., Hoffman, A., & Giovannini, E. (2005). *Handbook on constructing composite indicators: Methodology and user guide* (OECD Statistics Working Papers, No. 2005/03). OECD Publishing. https://doi.org/10.1787/533411815016

OECD, European Union, & European Commission Joint Research Centre. (2008). *Handbook on constructing composite indicators: Methodology and user guide*. OECD Publishing. https://doi.org/10.1787/9789264043466-en

Opitz, J. (2024). A closer look at classification evaluation metrics and a critical reflection of common evaluation practice in NLP. *Transactions of the Association for Computational Linguistics, 12*, 820–836. https://doi.org/10.1162/tacl_a_00675

Ravallion, M. (2012). Mashup indices of development. *The World Bank Research Observer, 27*(1), 1–32. https://doi.org/10.1093/wbro/lkr009

Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics, 20*, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7

Saisana, M., Saltelli, A., & Tarantola, S. (2005). Uncertainty and sensitivity analysis techniques as tools for the quality assessment of composite indicators. *Journal of the Royal Statistical Society Series A: Statistics in Society, 168*(2), 307–323. https://doi.org/10.1111/j.1467-985X.2005.00350.x

Saltelli, A. (2007). Composite indicators between analysis and advocacy. *Social Indicators Research, 81*(1), 65–77. https://doi.org/10.1007/s11205-006-0024-9

scikit-learn developers. (n.d.-a). *Decision trees*. scikit-learn. Retrieved November 19, 2026, from https://scikit-learn.org/stable/modules/tree.html#classification-criteria

scikit-learn developers. (n.d.-b). *User guide*. scikit-learn. Retrieved November 19, 2026, from https://scikit-learn.org/stable/user_guide.html

Serajuddin, U., Uematsu, H., Wieser, C., Yoshida, N., & Dabalen, A. (2015). *Data deprivation: Another deprivation to end* (Policy Research Working Paper No. 7252). World Bank. https://doi.org/10.1596/1813-9450-7252

Spearman, C. (1904). The proof and measurement of association between two things. *The American Journal of Psychology, 15*(1), 72–101. https://doi.org/10.2307/1412159

Troyanskaya, O., Cantor, M., Sherlock, G., Brown, P., Hastie, T., Tibshirani, R., Botstein, D., & Altman, R. B. (2001). Missing value estimation methods for DNA microarrays. *Bioinformatics, 17*(6), 520–525. https://doi.org/10.1093/bioinformatics/17.6.520

Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics, 7*, Article 91. https://doi.org/10.1186/1471-2105-7-91

World Bank. (n.d.). *How does the World Bank classify countries?* World Bank Data Help Desk. Retrieved November 19, 2026, from https://datahelpdesk.worldbank.org/knowledgebase/articles/378834-how-does-the-world-bank-classify-countries

World Bank. (2026). *World development indicators* [Data set]. Retrieved November 19, 2026, from https://datatopics.worldbank.org/world-development-indicators/

Zou, H., Hastie, T., & Tibshirani, R. (2006). Sparse principal component analysis. *Journal of Computational and Graphical Statistics, 15*(2), 265–286. https://doi.org/10.1198/106186006X113430

## Appendix A Indicator definitions and transformations

The following menu defines the input variables used in this study. Variable codes identify the official series for verification and are confined to this appendix. Positive magnitude variables marked Yes are transformed with log(1 + x) before averaging. The other variables remain on their recorded scale until clipping and standardisation. A high component loading does not establish that an indicator is normatively desirable.

| **Indicator**            | **Official series code** | **Log transform** |
|--------------------------|--------------------------|-------------------|
| GDP per capita           | NY.GDP.PCAP.CD           | Yes               |
| GNI per capita PPP       | NY.GNP.PCAP.PP.CD        | Yes               |
| GDP growth               | NY.GDP.MKTP.KD.ZG        | No                |
| Trade openness           | NE.TRD.GNFS.ZS           | No                |
| Unemployment             | SL.UEM.TOTL.ZS           | No                |
| FDI inflow               | BX.KLT.DINV.WD.GD.ZS     | No                |
| Life expectancy          | SP.DYN.LE00.IN           | No                |
| Under-5 mortality        | SH.DYN.MORT              | Yes               |
| Maternal mortality       | SH.STA.MMRT              | Yes               |
| Secondary enrolment      | SE.SEC.ENRR              | No                |
| Tertiary enrolment       | SE.TER.ENRR              | No                |
| Health spending          | SH.XPD.CHEX.GD.ZS        | No                |
| Measles immunisation     | SH.IMM.MEAS              | No                |
| Urbanisation             | SP.URB.TOTL.IN.ZS        | No                |
| Energy use per capita    | EG.USE.PCAP.KG.OE        | Yes               |
| Electricity access       | EG.ELC.ACCS.ZS           | No                |
| Renewable energy share   | EG.FEC.RNEW.ZS           | No                |
| Renewable electricity    | EG.ELC.RNEW.ZS           | No                |
| CO2 emissions per capita | EN.GHG.CO2.PC.CE.AR5     | Yes               |
| PM2.5 exposure           | EN.ATM.PM25.MC.M3        | Yes               |
| Forest cover             | AG.LND.FRST.ZS           | No                |
| Water withdrawal         | ER.H2O.FWTL.ZS           | Yes               |
| Agricultural land        | AG.LND.AGRI.ZS           | No                |
| Internet users           | IT.NET.USER.ZS           | No                |
| Mobile subscriptions     | IT.CEL.SETS.P2           | Yes               |
| Fixed broadband          | IT.NET.BBND.P2           | Yes               |
| Cereal yield             | AG.YLD.CREL.KG           | Yes               |
| Undernourishment         | SN.ITK.DEFC.ZS           | No                |

Table A1. The 28 selected official development series. Definitions and units follow the accompanying World Development Indicators series metadata.

## Statement of Individual Contribution

The three authors contributed equally. The work ran in four streams, and every author took part in all four. The streams were (i) problem framing and literature, (ii) data construction and code, (iii) experiments and robustness checks, (iv) writing, figures and review. The table records the primary owner of each stream. The percentage is the share of total effort.

| Member | Student ID | Primary responsibility | Contribution |
|---|---|---|---|
| Chen Wanchuan | 26108204G | Problem framing, index construction and stability analysis, and Sections 1, 3 and 6 | 33.4% |
| Lu Yuqing | 26079015G | Data pipeline, imputation and coverage analysis, the two shortlist procedures, and Sections 2 and 3 | 33.3% |
| Yuen Lok Yan | 26002016G | Classification, archetypes and dynamics, robustness checks, figure and table production, and Sections 4 and 5 | 33.3% |

We confirm that all three authors reviewed the complete analysis, re-ran the pipeline end to end on the released code, and checked every reported number against the artefacts in `results/`. Drafting of code and text was assisted by an AI coding tool. All outputs were verified against the released artefacts by the authors, who take full responsibility for the content.

Declaration. The data are public (World Bank WDI, CC BY 4.0) and were retrieved through the official API. No proprietary or personal data were used. The code, the intermediate artefacts and the exact commands to regenerate every table and figure accompany this submission.

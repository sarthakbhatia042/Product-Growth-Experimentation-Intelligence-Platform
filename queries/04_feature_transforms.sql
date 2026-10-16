-- 04_feature_transforms.sql
-- Covariate balance check: computes means, standard deviations, and standardized mean difference (SMD)
-- In a valid randomized experiment, |SMD| should be < 0.05 for all pre-treatment covariates.

SELECT
    'f0' AS feature,
    AVG(CASE WHEN treatment = 1 THEN f0 END) AS mean_treatment,
    AVG(CASE WHEN treatment = 0 THEN f0 END) AS mean_control,
    STDDEV(CASE WHEN treatment = 1 THEN f0 END) AS std_treatment,
    STDDEV(CASE WHEN treatment = 0 THEN f0 END) AS std_control,
    ABS(AVG(CASE WHEN treatment = 1 THEN f0 END) - AVG(CASE WHEN treatment = 0 THEN f0 END)) /
        SQRT((POWER(STDDEV(CASE WHEN treatment = 1 THEN f0 END), 2) + POWER(STDDEV(CASE WHEN treatment = 0 THEN f0 END), 2)) / 2.0) AS smd
FROM criteo_data
UNION ALL
SELECT
    'f1',
    AVG(CASE WHEN treatment = 1 THEN f1 END),
    AVG(CASE WHEN treatment = 0 THEN f1 END),
    STDDEV(CASE WHEN treatment = 1 THEN f1 END),
    STDDEV(CASE WHEN treatment = 0 THEN f1 END),
    ABS(AVG(CASE WHEN treatment = 1 THEN f1 END) - AVG(CASE WHEN treatment = 0 THEN f1 END)) /
        SQRT((POWER(STDDEV(CASE WHEN treatment = 1 THEN f1 END), 2) + POWER(STDDEV(CASE WHEN treatment = 0 THEN f1 END), 2)) / 2.0)
FROM criteo_data
UNION ALL
SELECT
    'f2',
    AVG(CASE WHEN treatment = 1 THEN f2 END),
    AVG(CASE WHEN treatment = 0 THEN f2 END),
    STDDEV(CASE WHEN treatment = 1 THEN f2 END),
    STDDEV(CASE WHEN treatment = 0 THEN f2 END),
    ABS(AVG(CASE WHEN treatment = 1 THEN f2 END) - AVG(CASE WHEN treatment = 0 THEN f2 END)) /
        SQRT((POWER(STDDEV(CASE WHEN treatment = 1 THEN f2 END), 2) + POWER(STDDEV(CASE WHEN treatment = 0 THEN f2 END), 2)) / 2.0)
FROM criteo_data
UNION ALL
SELECT
    'f3',
    AVG(CASE WHEN treatment = 1 THEN f3 END),
    AVG(CASE WHEN treatment = 0 THEN f3 END),
    STDDEV(CASE WHEN treatment = 1 THEN f3 END),
    STDDEV(CASE WHEN treatment = 0 THEN f3 END),
    ABS(AVG(CASE WHEN treatment = 1 THEN f3 END) - AVG(CASE WHEN treatment = 0 THEN f3 END)) /
        SQRT((POWER(STDDEV(CASE WHEN treatment = 1 THEN f3 END), 2) + POWER(STDDEV(CASE WHEN treatment = 0 THEN f3 END), 2)) / 2.0)
FROM criteo_data;

-- 03_srm_check.sql
-- Performs Sample Ratio Mismatch (SRM) Chi-Squared goodness-of-fit test
-- Compares observed cohort counts against expected nominal allocation (e.g. 84.6% Treatment : 15.4% Control)

WITH cohort_counts AS (
    SELECT
        COUNT(*) AS total_sample,
        SUM(CASE WHEN treatment = 1 THEN 1 ELSE 0 END) AS observed_treatment,
        SUM(CASE WHEN treatment = 0 THEN 1 ELSE 0 END) AS observed_control
    FROM criteo_data
),
expected_calc AS (
    SELECT
        total_sample,
        observed_treatment,
        observed_control,
        total_sample * 0.8460 AS expected_treatment,
        total_sample * 0.1540 AS expected_control
    FROM cohort_counts
)
SELECT
    total_sample,
    observed_treatment,
    expected_treatment,
    observed_control,
    expected_control,
    -- Chi-square components: (O - E)^2 / E
    POWER(observed_treatment - expected_treatment, 2) / expected_treatment AS chi2_treatment,
    POWER(observed_control - expected_control, 2) / expected_control AS chi2_control,
    (POWER(observed_treatment - expected_treatment, 2) / expected_treatment) +
    (POWER(observed_control - expected_control, 2) / expected_control) AS chi2_stat,
    1 AS df
FROM expected_calc;

-- 01_data_quality_audit.sql
-- Performs schema integrity, missingness audit, and domain constraints check on Criteo dataset

WITH basic_checks AS (
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT hash(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, treatment, exposure, visit, conversion)) AS distinct_records,
        SUM(CASE WHEN f0 IS NULL THEN 1 ELSE 0 END) AS null_f0,
        SUM(CASE WHEN f1 IS NULL THEN 1 ELSE 0 END) AS null_f1,
        SUM(CASE WHEN f2 IS NULL THEN 1 ELSE 0 END) AS null_f2,
        SUM(CASE WHEN f3 IS NULL THEN 1 ELSE 0 END) AS null_f3,
        SUM(CASE WHEN f4 IS NULL THEN 1 ELSE 0 END) AS null_f4,
        SUM(CASE WHEN f5 IS NULL THEN 1 ELSE 0 END) AS null_f5,
        SUM(CASE WHEN f6 IS NULL THEN 1 ELSE 0 END) AS null_f6,
        SUM(CASE WHEN f7 IS NULL THEN 1 ELSE 0 END) AS null_f7,
        SUM(CASE WHEN f8 IS NULL THEN 1 ELSE 0 END) AS null_f8,
        SUM(CASE WHEN f9 IS NULL THEN 1 ELSE 0 END) AS null_f9,
        SUM(CASE WHEN f10 IS NULL THEN 1 ELSE 0 END) AS null_f10,
        SUM(CASE WHEN f11 IS NULL THEN 1 ELSE 0 END) AS null_f11,
        SUM(CASE WHEN treatment IS NULL THEN 1 ELSE 0 END) AS null_treatment,
        SUM(CASE WHEN exposure IS NULL THEN 1 ELSE 0 END) AS null_exposure,
        SUM(CASE WHEN visit IS NULL THEN 1 ELSE 0 END) AS null_visit,
        SUM(CASE WHEN conversion IS NULL THEN 1 ELSE 0 END) AS null_conversion,
        -- Value domain integrity checks
        SUM(CASE WHEN treatment NOT IN (0, 1) THEN 1 ELSE 0 END) AS invalid_treatment_values,
        SUM(CASE WHEN exposure NOT IN (0, 1) THEN 1 ELSE 0 END) AS invalid_exposure_values,
        SUM(CASE WHEN visit NOT IN (0, 1) THEN 1 ELSE 0 END) AS invalid_visit_values,
        SUM(CASE WHEN conversion NOT IN (0, 1) THEN 1 ELSE 0 END) AS invalid_conversion_values,
        -- Logical consistency: can control users be exposed? Should be 0 by definition
        SUM(CASE WHEN treatment = 0 AND exposure = 1 THEN 1 ELSE 0 END) AS control_leakage_exposure
    FROM criteo_data
)
SELECT * FROM basic_checks;

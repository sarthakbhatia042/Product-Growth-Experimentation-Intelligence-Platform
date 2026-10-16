-- 02_cohort_summary.sql
-- Summarizes conversion rates, visit rates, and exposure distribution by treatment assignment

SELECT
    treatment,
    CASE WHEN treatment = 1 THEN 'Treatment' ELSE 'Control' END AS cohort_name,
    COUNT(*) AS cohort_size,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 4) AS cohort_pct,
    SUM(exposure) AS exposures_count,
    ROUND(AVG(exposure) * 100.0, 4) AS exposure_rate_pct,
    SUM(visit) AS visits_count,
    ROUND(AVG(visit) * 100.0, 4) AS visit_rate_pct,
    SUM(conversion) AS conversions_count,
    ROUND(AVG(conversion) * 100.0, 4) AS conversion_rate_pct,
    -- Conversion rate given visit (conversion funnel conditional rate)
    ROUND(SUM(conversion) * 100.0 / NULLIF(SUM(visit), 0), 4) AS conv_per_visit_pct
FROM criteo_data
GROUP BY treatment
ORDER BY treatment DESC;

\set ON_ERROR_STOP 1
SET statement_timeout = '600000ms';
SET default_transaction_read_only = off;

CREATE TEMP TABLE frozen_mimic (
  subject_id integer,
  hadm_id integer,
  stay_id integer,
  intime timestamp,
  outtime timestamp,
  age_at_icu numeric,
  gender text,
  alignment_tier text,
  death28 integer
);

\copy frozen_mimic FROM 'D:/Clinical_Public_Database_Research/01_ICU_Research/01_Active_Projects/POSTOP_ABDOMINAL_TEMPORAL_LITE_20260918/04_formal_modeling/02_data_internal/mimic_formal_cohort.csv' CSV HEADER

CREATE TEMP TABLE surgery_map (
  icd_version integer,
  icd_code text,
  long_title text,
  surgery_family text
);

\copy surgery_map FROM 'D:/Clinical_Public_Database_Research/01_ICU_Research/01_Active_Projects/POSTOP_ABDOMINAL_TEMPORAL_LITE_20260918/03_stage0/flow_outputs/mimiciv_surgery_code_map.csv' CSV HEADER

CREATE TEMP TABLE mimic_feature_export AS
WITH indexed AS (
  SELECT f.*, 'M' || lpad(row_number() OVER (ORDER BY stay_id)::text, 6, '0') AS analysis_id
  FROM frozen_mimic f
), surgery AS (
  SELECT f.stay_id,
         string_agg(DISTINCT sm.surgery_family, ';' ORDER BY sm.surgery_family) AS surgery_families,
         string_agg(DISTINCT (p.icd_version::text || ':' || trim(p.icd_code)), ';' ORDER BY (p.icd_version::text || ':' || trim(p.icd_code))) AS surgery_codes
  FROM frozen_mimic f
  JOIN mimiciv_hosp.procedures_icd p
    ON p.subject_id=f.subject_id AND p.hadm_id=f.hadm_id
   AND p.chartdate BETWEEN f.intime::date - 1 AND f.intime::date
  JOIN surgery_map sm ON sm.icd_version=p.icd_version AND sm.icd_code=trim(p.icd_code)
  GROUP BY f.stay_id
), htn AS (
  SELECT f.hadm_id,
         bool_or(
           (d.icd_version=9 AND (trim(d.icd_code) LIKE '401%' OR trim(d.icd_code) LIKE '402%' OR trim(d.icd_code) LIKE '403%' OR trim(d.icd_code) LIKE '404%' OR trim(d.icd_code) LIKE '405%'))
           OR
           (d.icd_version=10 AND (trim(d.icd_code) LIKE 'I10%' OR trim(d.icd_code) LIKE 'I11%' OR trim(d.icd_code) LIKE 'I12%' OR trim(d.icd_code) LIKE 'I13%' OR trim(d.icd_code) LIKE 'I15%'))
         )::integer AS hypertension
  FROM frozen_mimic f
  LEFT JOIN mimiciv_hosp.diagnoses_icd d ON d.subject_id=f.subject_id AND d.hadm_id=f.hadm_id
  GROUP BY f.hadm_id
), labs_long AS (
  SELECT f.stay_id,
         CASE
           WHEN le.itemid=50813 THEN 'lactate'
           WHEN le.itemid=50912 THEN 'creatinine'
           WHEN le.itemid=51222 THEN 'hemoglobin'
           WHEN le.itemid IN (51300,51301) THEN 'wbc'
           WHEN le.itemid=51265 THEN 'platelets'
         END AS analyte,
         le.charttime,
         le.itemid,
         le.valuenum,
         le.valueuom,
         row_number() OVER (
           PARTITION BY f.stay_id,
             CASE
               WHEN le.itemid=50813 THEN 'lactate'
               WHEN le.itemid=50912 THEN 'creatinine'
               WHEN le.itemid=51222 THEN 'hemoglobin'
               WHEN le.itemid IN (51300,51301) THEN 'wbc'
               WHEN le.itemid=51265 THEN 'platelets'
             END
           ORDER BY le.charttime, le.itemid
         ) AS rn
  FROM frozen_mimic f
  JOIN mimiciv_hosp.labevents le ON le.hadm_id=f.hadm_id
   AND le.charttime>=f.intime AND le.charttime<f.intime+interval '12 hour'
  WHERE le.itemid IN (50813,50912,51222,51265,51300,51301)
    AND le.valuenum IS NOT NULL
    AND (
      (le.itemid=50813 AND le.valueuom='mmol/L' AND le.valuenum BETWEEN 0.1 AND 40)
      OR (le.itemid=50912 AND le.valueuom='mg/dL' AND le.valuenum BETWEEN 0.1 AND 30)
      OR (le.itemid=51222 AND le.valueuom='g/dL' AND le.valuenum BETWEEN 1 AND 25)
      OR (le.itemid IN (51300,51301) AND le.valueuom='K/uL' AND le.valuenum BETWEEN 0.01 AND 300)
      OR (le.itemid=51265 AND le.valueuom='K/uL' AND le.valuenum BETWEEN 1 AND 2000)
    )
), labs AS (
  SELECT stay_id,
         max(valuenum) FILTER (WHERE analyte='lactate' AND rn=1) AS lactate,
         max(itemid) FILTER (WHERE analyte='lactate' AND rn=1) AS lactate_itemid,
         max(valueuom) FILTER (WHERE analyte='lactate' AND rn=1) AS lactate_unit,
         max(charttime) FILTER (WHERE analyte='lactate' AND rn=1) AS lactate_time,
         max(valuenum) FILTER (WHERE analyte='creatinine' AND rn=1) AS creatinine,
         max(itemid) FILTER (WHERE analyte='creatinine' AND rn=1) AS creatinine_itemid,
         max(valueuom) FILTER (WHERE analyte='creatinine' AND rn=1) AS creatinine_unit,
         max(charttime) FILTER (WHERE analyte='creatinine' AND rn=1) AS creatinine_time,
         max(valuenum) FILTER (WHERE analyte='hemoglobin' AND rn=1) AS hemoglobin,
         max(itemid) FILTER (WHERE analyte='hemoglobin' AND rn=1) AS hemoglobin_itemid,
         max(valueuom) FILTER (WHERE analyte='hemoglobin' AND rn=1) AS hemoglobin_unit,
         max(charttime) FILTER (WHERE analyte='hemoglobin' AND rn=1) AS hemoglobin_time,
         max(valuenum) FILTER (WHERE analyte='wbc' AND rn=1) AS wbc,
         max(itemid) FILTER (WHERE analyte='wbc' AND rn=1) AS wbc_itemid,
         max(valueuom) FILTER (WHERE analyte='wbc' AND rn=1) AS wbc_unit,
         max(charttime) FILTER (WHERE analyte='wbc' AND rn=1) AS wbc_time,
         max(valuenum) FILTER (WHERE analyte='platelets' AND rn=1) AS platelets,
         max(itemid) FILTER (WHERE analyte='platelets' AND rn=1) AS platelets_itemid,
         max(valueuom) FILTER (WHERE analyte='platelets' AND rn=1) AS platelets_unit,
         max(charttime) FILTER (WHERE analyte='platelets' AND rn=1) AS platelets_time
  FROM labs_long WHERE rn=1 GROUP BY stay_id
)
SELECT f.analysis_id,f.subject_id,f.hadm_id,f.stay_id,f.intime,f.outtime,
       f.age_at_icu AS age_years,
       CASE WHEN f.gender='M' THEN 1 WHEN f.gender='F' THEN 0 END AS male,
       f.death28,
       f.alignment_tier,
       CASE WHEN s.surgery_families IS NULL THEN 'other_unclassifiable'
            WHEN position(';' in s.surgery_families)>0 THEN 'multiple_qualifying_families'
            ELSE s.surgery_families END AS surgery_family,
       s.surgery_codes,
       a.admission_type AS urgency_source_value,
       CASE WHEN a.admission_type IN ('EW EMER.','DIRECT EMER.','URGENT') THEN 1
            WHEN a.admission_type IN ('ELECTIVE','SURGICAL SAME DAY ADMISSION') THEN 0 END AS urgent_or_emergency,
       CASE WHEN ht.height_cm BETWEEN 100 AND 250 THEN ht.height_cm END AS height_cm,
       CASE WHEN wt.weight_kg BETWEEN 20 AND 400 THEN wt.weight_kg END AS weight_kg,
       CASE WHEN ht.height_cm BETWEEN 100 AND 250 AND wt.weight_kg BETWEEN 20 AND 400
            THEN wt.weight_kg/power(ht.height_cm/100.0,2) END AS bmi,
       coalesce(h.hypertension,0) AS hypertension,
       CASE WHEN coalesce(c.diabetes_without_cc,0)=1 OR coalesce(c.diabetes_with_cc,0)=1 THEN 1 ELSE 0 END AS diabetes,
       coalesce(c.chronic_pulmonary_disease,0) AS chronic_lung_disease,
       coalesce(c.renal_disease,0) AS chronic_renal_disease,
       coalesce(c.congestive_heart_failure,0) AS congestive_heart_failure,
       CASE WHEN coalesce(c.malignant_cancer,0)=1 OR coalesce(c.metastatic_solid_tumor,0)=1 THEN 1 ELSE 0 END AS malignancy,
       c.charlson_comorbidity_index,
       l.lactate,l.lactate_itemid,l.lactate_unit,
       extract(epoch from (l.lactate_time-f.intime))/60.0 AS lactate_offset_min,
       l.creatinine,l.creatinine_itemid,l.creatinine_unit,
       extract(epoch from (l.creatinine_time-f.intime))/60.0 AS creatinine_offset_min,
       l.hemoglobin,l.hemoglobin_itemid,l.hemoglobin_unit,
       extract(epoch from (l.hemoglobin_time-f.intime))/60.0 AS hemoglobin_offset_min,
       l.wbc,l.wbc_itemid,l.wbc_unit,
       extract(epoch from (l.wbc_time-f.intime))/60.0 AS wbc_offset_min,
       l.platelets,l.platelets_itemid,l.platelets_unit,
       extract(epoch from (l.platelets_time-f.intime))/60.0 AS platelets_offset_min,
       EXISTS (
         SELECT 1 FROM mimiciv_derived.ventilation v
         WHERE v.stay_id=f.stay_id AND v.starttime<f.intime+interval '12 hour' AND v.endtime>f.intime
           AND v.ventilation_status IN ('InvasiveVent','Tracheostomy')
       )::integer AS invasive_mechanical_ventilation,
       EXISTS (
         SELECT 1 FROM mimiciv_derived.vasoactive_agent v
         WHERE v.stay_id=f.stay_id AND v.starttime<f.intime+interval '12 hour' AND v.endtime>f.intime
           AND (coalesce(v.dopamine,0)>0 OR coalesce(v.epinephrine,0)>0 OR coalesce(v.norepinephrine,0)>0 OR coalesce(v.phenylephrine,0)>0 OR coalesce(v.vasopressin,0)>0)
       )::integer AS any_vasopressor
FROM indexed f
JOIN mimiciv_hosp.admissions a ON a.hadm_id=f.hadm_id
LEFT JOIN surgery s ON s.stay_id=f.stay_id
LEFT JOIN htn h ON h.hadm_id=f.hadm_id
LEFT JOIN mimiciv_derived.charlson c ON c.subject_id=f.subject_id AND c.hadm_id=f.hadm_id
LEFT JOIN labs l ON l.stay_id=f.stay_id
LEFT JOIN LATERAL (
  SELECT x.height::double precision AS height_cm
  FROM mimiciv_derived.height x
  WHERE x.stay_id=f.stay_id AND x.charttime<f.intime+interval '12 hour'
  ORDER BY abs(extract(epoch from (x.charttime-f.intime))),x.charttime LIMIT 1
) ht ON true
LEFT JOIN LATERAL (
  SELECT x.weight AS weight_kg
  FROM mimiciv_derived.weight_durations x
  WHERE x.stay_id=f.stay_id AND x.starttime<f.intime+interval '12 hour' AND x.endtime>f.intime
  ORDER BY CASE WHEN x.weight_type='admit' THEN 0 ELSE 1 END,x.starttime LIMIT 1
) wt ON true
ORDER BY f.stay_id;

\copy mimic_feature_export TO 'D:/Clinical_Public_Database_Research/01_ICU_Research/01_Active_Projects/POSTOP_ABDOMINAL_TEMPORAL_LITE_20260918/10_table1_work/raw/mimic_table1_raw.csv' CSV HEADER

CREATE TEMP TABLE mimic_lab_audit_export AS
SELECT le.itemid,di.label,di.fluid,di.category,le.valueuom,
       count(*) AS row_count,count(DISTINCT f.stay_id) AS patient_count,
       min(le.valuenum) AS min_value,max(le.valuenum) AS max_value,
       min(extract(epoch from (le.charttime-f.intime))/60.0) AS min_offset_min,
       max(extract(epoch from (le.charttime-f.intime))/60.0) AS max_offset_min
FROM frozen_mimic f
JOIN mimiciv_hosp.labevents le ON le.hadm_id=f.hadm_id
 AND le.charttime>=f.intime AND le.charttime<f.intime+interval '12 hour'
JOIN mimiciv_hosp.d_labitems di ON di.itemid=le.itemid
WHERE le.itemid IN (50811,50813,50912,51222,51265,51300,51301,51640,51755,51756,52442,52546,53154,53189)
  AND le.valuenum IS NOT NULL
GROUP BY 1,2,3,4,5 ORDER BY 1,5;

\copy mimic_lab_audit_export TO 'D:/Clinical_Public_Database_Research/01_ICU_Research/01_Active_Projects/POSTOP_ABDOMINAL_TEMPORAL_LITE_20260918/10_table1_work/raw/mimic_lab_candidate_audit.csv' CSV HEADER


\set ON_ERROR_STOP 1
SET statement_timeout = '600000ms';
SET default_transaction_read_only = off;

CREATE TEMP TABLE frozen_sicdb (
  caseid bigint,
  patientid bigint,
  icuoffset bigint,
  timeofstay bigint,
  age_on_admission numeric,
  sex_code bigint,
  sex_label text,
  death28 integer
);

\copy frozen_sicdb FROM 'D:/Clinical_Public_Database_Research/01_ICU_Research/01_Active_Projects/POSTOP_ABDOMINAL_TEMPORAL_LITE_20260918/04_formal_modeling/02_data_internal/sicdb_formal_cohort.csv' CSV HEADER

CREATE TEMP TABLE sicdb_feature_export AS
WITH indexed AS (
  SELECT f.*, 'S' || lpad(row_number() OVER (ORDER BY caseid)::text, 6, '0') AS analysis_id
  FROM frozen_sicdb f
), preconditions AS (
  SELECT f.caseid,
         max((dr."RefID"=740)::integer) FILTER (WHERE dr."FieldID"=3097) AS diabetes,
         max((dr."RefID"=740)::integer) FILTER (WHERE dr."FieldID"=3098) AS hypertension,
         max((dr."RefID"=740)::integer) FILTER (WHERE dr."FieldID"=3103) AS chronic_lung_disease,
         max((dr."RefID"=740)::integer) FILTER (WHERE dr."FieldID"=3104) AS chronic_renal_disease
  FROM frozen_sicdb f
  LEFT JOIN data_ref dr ON dr."CaseID"=f.caseid AND dr."FieldID" IN (3097,3098,3103,3104)
  GROUP BY f.caseid
), labs_long AS (
  SELECT f.caseid,
         CASE
           WHEN l."LaboratoryID" IN (465,657) THEN 'lactate'
           WHEN l."LaboratoryID"=367 THEN 'creatinine'
           WHEN l."LaboratoryID"=289 THEN 'hemoglobin'
           WHEN l."LaboratoryID"=301 THEN 'wbc'
           WHEN l."LaboratoryID" IN (314,315) THEN 'platelets'
         END AS analyte,
         l."Offset" AS lab_offset,
         l."LaboratoryID" AS itemid,
         l."LaboratoryValue" AS value,
         d."ReferenceUnit" AS unit,
         row_number() OVER (
           PARTITION BY f.caseid,
             CASE
               WHEN l."LaboratoryID" IN (465,657) THEN 'lactate'
               WHEN l."LaboratoryID"=367 THEN 'creatinine'
               WHEN l."LaboratoryID"=289 THEN 'hemoglobin'
               WHEN l."LaboratoryID"=301 THEN 'wbc'
               WHEN l."LaboratoryID" IN (314,315) THEN 'platelets'
             END
           ORDER BY l."Offset",l."LaboratoryID"
         ) AS rn
  FROM frozen_sicdb f
  JOIN laboratory l ON l."CaseID"=f.caseid
   AND l."Offset">=f.icuoffset AND l."Offset"<f.icuoffset+43200
  JOIN d_references d ON d."ReferenceGlobalID"=l."LaboratoryID"
  WHERE l."LaboratoryID" IN (289,301,314,315,367,465,657)
    AND l."LaboratoryValue" IS NOT NULL
    AND (
      (l."LaboratoryID" IN (465,657) AND d."ReferenceUnit"='mmol/L' AND l."LaboratoryValue" BETWEEN 0.1 AND 40)
      OR (l."LaboratoryID"=367 AND lower(d."ReferenceUnit")='mg/dl' AND l."LaboratoryValue" BETWEEN 0.1 AND 30)
      OR (l."LaboratoryID"=289 AND lower(d."ReferenceUnit")='g/dl' AND l."LaboratoryValue" BETWEEN 1 AND 25)
      OR (l."LaboratoryID"=301 AND d."ReferenceUnit"='g/L' AND l."LaboratoryValue" BETWEEN 0.01 AND 300)
      OR (l."LaboratoryID" IN (314,315) AND d."ReferenceUnit"='g/L' AND l."LaboratoryValue" BETWEEN 1 AND 2000)
    )
), labs AS (
  SELECT caseid,
         max(value) FILTER (WHERE analyte='lactate' AND rn=1) AS lactate,
         max(itemid) FILTER (WHERE analyte='lactate' AND rn=1) AS lactate_itemid,
         max(unit) FILTER (WHERE analyte='lactate' AND rn=1) AS lactate_unit,
         max(lab_offset) FILTER (WHERE analyte='lactate' AND rn=1) AS lactate_time,
         max(value) FILTER (WHERE analyte='creatinine' AND rn=1) AS creatinine,
         max(itemid) FILTER (WHERE analyte='creatinine' AND rn=1) AS creatinine_itemid,
         max(unit) FILTER (WHERE analyte='creatinine' AND rn=1) AS creatinine_unit,
         max(lab_offset) FILTER (WHERE analyte='creatinine' AND rn=1) AS creatinine_time,
         max(value) FILTER (WHERE analyte='hemoglobin' AND rn=1) AS hemoglobin,
         max(itemid) FILTER (WHERE analyte='hemoglobin' AND rn=1) AS hemoglobin_itemid,
         max(unit) FILTER (WHERE analyte='hemoglobin' AND rn=1) AS hemoglobin_unit,
         max(lab_offset) FILTER (WHERE analyte='hemoglobin' AND rn=1) AS hemoglobin_time,
         max(value) FILTER (WHERE analyte='wbc' AND rn=1) AS wbc,
         max(itemid) FILTER (WHERE analyte='wbc' AND rn=1) AS wbc_itemid,
         max(unit) FILTER (WHERE analyte='wbc' AND rn=1) AS wbc_unit,
         max(lab_offset) FILTER (WHERE analyte='wbc' AND rn=1) AS wbc_time,
         max(value) FILTER (WHERE analyte='platelets' AND rn=1) AS platelets,
         max(itemid) FILTER (WHERE analyte='platelets' AND rn=1) AS platelets_itemid,
         max(unit) FILTER (WHERE analyte='platelets' AND rn=1) AS platelets_unit,
         max(lab_offset) FILTER (WHERE analyte='platelets' AND rn=1) AS platelets_time
  FROM labs_long WHERE rn=1 GROUP BY caseid
), airway_evidence AS (
  SELECT DISTINCT f.caseid
  FROM frozen_sicdb f
  JOIN data_range ar ON ar."CaseID"=f.caseid AND ar."DataID" IN (720,3041)
   AND ar."Offset"<f.icuoffset+43200 AND coalesce(ar."OffsetEnd",ar."Offset")>f.icuoffset
), respirator_evidence AS (
  SELECT DISTINCT f.caseid
  FROM frozen_sicdb f
  JOIN data_float_h rs ON rs."CaseID"=f.caseid
   AND rs."DataID" IN (2278,2282,2283,2284,2285,3040,3121)
   AND rs."Offset">=f.icuoffset AND rs."Offset"<f.icuoffset+43200 AND rs."Val" IS NOT NULL
), vasopressor_evidence AS (
  SELECT DISTINCT f.caseid
  FROM frozen_sicdb f
  JOIN medication m ON m."CaseID"=f.caseid AND m."DrugID" IN (1502,1550,1562,1593,1618)
  WHERE (
      (m."IsSingleDose"=1 AND m."Offset">=f.icuoffset AND m."Offset"<f.icuoffset+43200)
      OR
      (coalesce(m."IsSingleDose",0)<>1 AND m."Offset"<f.icuoffset+43200 AND coalesce(m."OffsetDrugEnd",m."Offset")>f.icuoffset)
    )
    AND (coalesce(m."Amount",0)>0 OR coalesce(m."AmountPerMinute",0)>0)
)
SELECT f.analysis_id,f.caseid,f.patientid,f.icuoffset,f.timeofstay,
       f.age_on_admission AS age_years,
       CASE WHEN lower(f.sex_label)='male' THEN 1 WHEN lower(f.sex_label)='female' THEN 0 END AS male,
       f.death28,
       CASE c."SurgicalSite"
         WHEN 2211 THEN 'intestinal_colorectal'
         WHEN 2219 THEN 'upper_gi'
         WHEN 2225 THEN 'hepatobiliary_pancreatic'
         WHEN 2242 THEN 'hepatobiliary_pancreatic'
         WHEN 2253 THEN 'hepatobiliary_pancreatic'
         ELSE 'other_unclassifiable' END AS surgery_family,
       c."SurgicalSite" AS surgery_site_id,
       ss."ReferenceValue" AS surgery_site_label,
       c."AdmissionUrgency" AS urgency_source_id,
       au."ReferenceValue" AS urgency_source_value,
       CASE WHEN c."AdmissionUrgency"=3137 THEN 1 WHEN c."AdmissionUrgency"=3138 THEN 0 END AS urgent_or_emergency,
       CASE WHEN c."HeightOnAdmission" BETWEEN 100 AND 250 THEN c."HeightOnAdmission" END AS height_cm,
       CASE WHEN c."WeightOnAdmission" BETWEEN 20000 AND 400000 THEN c."WeightOnAdmission"/1000.0 END AS weight_kg,
       CASE WHEN c."HeightOnAdmission" BETWEEN 100 AND 250 AND c."WeightOnAdmission" BETWEEN 20000 AND 400000
            THEN (c."WeightOnAdmission"/1000.0)/power(c."HeightOnAdmission"/100.0,2) END AS bmi,
       p.hypertension,p.diabetes,p.chronic_lung_disease,p.chronic_renal_disease,
       l.lactate,l.lactate_itemid,l.lactate_unit,(l.lactate_time-f.icuoffset)/60.0 AS lactate_offset_min,
       l.creatinine,l.creatinine_itemid,l.creatinine_unit,(l.creatinine_time-f.icuoffset)/60.0 AS creatinine_offset_min,
       l.hemoglobin,l.hemoglobin_itemid,l.hemoglobin_unit,(l.hemoglobin_time-f.icuoffset)/60.0 AS hemoglobin_offset_min,
       l.wbc,l.wbc_itemid,l.wbc_unit,(l.wbc_time-f.icuoffset)/60.0 AS wbc_offset_min,
       l.platelets,l.platelets_itemid,l.platelets_unit,(l.platelets_time-f.icuoffset)/60.0 AS platelets_offset_min,
       (aw.caseid IS NOT NULL AND rs.caseid IS NOT NULL)::integer AS invasive_mechanical_ventilation,
       (vp.caseid IS NOT NULL)::integer AS any_vasopressor
FROM indexed f
JOIN cases c ON c."CaseID"=f.caseid
LEFT JOIN d_references ss ON ss."ReferenceGlobalID"=c."SurgicalSite"
LEFT JOIN d_references au ON au."ReferenceGlobalID"=c."AdmissionUrgency"
LEFT JOIN preconditions p ON p.caseid=f.caseid
LEFT JOIN labs l ON l.caseid=f.caseid
LEFT JOIN airway_evidence aw ON aw.caseid=f.caseid
LEFT JOIN respirator_evidence rs ON rs.caseid=f.caseid
LEFT JOIN vasopressor_evidence vp ON vp.caseid=f.caseid
ORDER BY f.caseid;

\copy sicdb_feature_export TO 'D:/Clinical_Public_Database_Research/01_ICU_Research/01_Active_Projects/POSTOP_ABDOMINAL_TEMPORAL_LITE_20260918/10_table1_work/raw/sicdb_table1_raw.csv' CSV HEADER

CREATE TEMP TABLE sicdb_lab_audit_export AS
SELECT l."LaboratoryID" AS itemid,d."ReferenceName" AS source_name,d."ReferenceValue" AS label,d."ReferenceUnit" AS unit,
       count(*) AS row_count,count(DISTINCT f.caseid) AS patient_count,
       min(l."LaboratoryValue") AS min_value,max(l."LaboratoryValue") AS max_value,
       min((l."Offset"-f.icuoffset)/60.0) AS min_offset_min,
       max((l."Offset"-f.icuoffset)/60.0) AS max_offset_min
FROM frozen_sicdb f
JOIN laboratory l ON l."CaseID"=f.caseid AND l."Offset">=f.icuoffset AND l."Offset"<f.icuoffset+43200
JOIN d_references d ON d."ReferenceGlobalID"=l."LaboratoryID"
WHERE l."LaboratoryID" IN (288,289,301,314,315,339,367,368,454,465,657,658)
  AND l."LaboratoryValue" IS NOT NULL
GROUP BY 1,2,3,4 ORDER BY 1;

\copy sicdb_lab_audit_export TO 'D:/Clinical_Public_Database_Research/01_ICU_Research/01_Active_Projects/POSTOP_ABDOMINAL_TEMPORAL_LITE_20260918/10_table1_work/raw/sicdb_lab_candidate_audit.csv' CSV HEADER

CREATE TEMP TABLE sicdb_treatment_audit_export AS
SELECT d."ReferenceGlobalID" AS source_id,d."ReferenceName" AS source_name,d."ReferenceValue" AS source_value,d."ReferenceUnit" AS source_unit,
       CASE
         WHEN d."ReferenceGlobalID" IN (720,3041) THEN 'airway_interval'
         WHEN d."ReferenceGlobalID" IN (2278,2282,2283,2284,2285,3040,3121) THEN 'respirator_setting'
         WHEN d."ReferenceGlobalID" IN (1502,1550,1562,1593,1618) THEN 'vasopressor_medication'
       END AS construct_role
FROM d_references d
WHERE d."ReferenceGlobalID" IN (720,3041,2278,2282,2283,2284,2285,3040,3121,1502,1550,1562,1593,1618)
ORDER BY 5,1;

\copy sicdb_treatment_audit_export TO 'D:/Clinical_Public_Database_Research/01_ICU_Research/01_Active_Projects/POSTOP_ABDOMINAL_TEMPORAL_LITE_20260918/10_table1_work/raw/sicdb_treatment_concept_audit.csv' CSV HEADER


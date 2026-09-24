[CmdletBinding()]
param(
    [SecureString]$PgPassword,
    [string]$PsqlPath = 'psql',
    [string]$PgHost = 'localhost',
    [string]$PgUser = 'postgres',
    [int]$PgPort = 5432
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Output = Join-Path $Root 'flow_outputs'
New-Item -ItemType Directory -Path $Output -Force | Out-Null
if (-not (Test-Path -LiteralPath $PsqlPath)) {
    $resolvedPsql = Get-Command $PsqlPath -ErrorAction SilentlyContinue
    if (-not $resolvedPsql) { throw "psql client not found: $PsqlPath" }
    $PsqlPath = $resolvedPsql.Source
}
if (-not $PgPassword) { $PgPassword = Read-Host 'PostgreSQL password (used only in this process)' -AsSecureString }
$Handle = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($PgPassword)
try { $PlainPassword = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($Handle) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($Handle) }

function Invoke-ReadOnlyCsv {
    param([string]$Database,[string]$Name,[string]$Sql)
    $psi = [Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $PsqlPath
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    foreach ($arg in @('-X','-w','-P','pager=off','-v','ON_ERROR_STOP=1','-h',$PgHost,'-p',([string]$PgPort),'-U',$PgUser,'-d',$Database,'--csv','-c',$Sql)) { [void]$psi.ArgumentList.Add($arg) }
    $process = [Diagnostics.Process]::new(); $process.StartInfo = $psi; [void]$process.Start()
    $stdout = $process.StandardOutput.ReadToEnd(); $stderr = $process.StandardError.ReadToEnd(); $process.WaitForExit()
    if ($process.ExitCode -ne 0) { throw "psql failed for $Database/$Name (exit $($process.ExitCode)): $stderr" }
    $Artifact = ("{0}_{1}.csv" -f $Database,$Name); $Path = Join-Path $Output $Artifact
    [IO.File]::WriteAllText($Path,$stdout,[Text.UTF8Encoding]::new($true))
    [pscustomobject]@{database=$Database;artifact=$Artifact;rows=[Math]::Max(0,($stdout -split "`n").Count-2);sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant();stderr=$stderr.Trim()}
}

$MimicCodeMap = @'
WITH code_map AS (
 SELECT d.icd_version,trim(d.icd_code) AS icd_code,d.long_title,
        CASE
          WHEN lower(d.long_title) ~ '(pancrea|liver|hepatic|hepat|gallbladder|biliar|cholecyst)' THEN 'hepatobiliary_pancreatic'
          WHEN lower(d.long_title) ~ '(esoph|stomach|gastr|duoden|jejun)' THEN 'upper_gi'
          WHEN lower(d.long_title) ~ '(small intestine|small bowel|intestinal|intestine|ileum|colon|cecum|colect|rectum|rectal|proctect)' THEN 'intestinal_colorectal'
        END AS surgery_family
 FROM mimiciv_hosp.d_icd_procedures d
 WHERE
   (
     d.icd_version=10
     AND lower(d.long_title) ~ '(pancrea|liver|hepatic|hepat|gallbladder|biliar|cholecyst|esoph|stomach|gastr|duoden|jejun|small intestine|small bowel|intestinal|intestine|ileum|colon|cecum|colect|rectum|rectal|proctect)'
     AND lower(d.long_title) ~ '(resection|excision|bypass|repair|replacement|supplement|revision)'
     AND lower(d.long_title) ~ '(open approach|percutaneous endoscopic approach)'
     AND lower(d.long_title) !~ '(diagnostic|transplant)'
   )
   OR
   (
     d.icd_version=9
     AND lower(d.long_title) ~ '(pancrea|liver|hepatic|hepat|gallbladder|biliar|cholecyst|esoph|stomach|gastr|duoden|jejun|small intestine|small bowel|intestinal|intestine|ileum|colon|cecum|colect|rectum|rectal|proctect)'
     AND lower(d.long_title) ~ '(excision|resection|esophagectomy|gastrectomy|colectomy|hemicolectomy|anastomosis|repair|bypass|hepatectomy|lobectomy|pancreatectomy|pancreaticoduodenectomy|cholecystectomy)'
     AND lower(d.long_title) !~ '(biopsy|endoscop|percutaneous|needle|aspiration|dilat|drainage|catheter|diagnostic|transplant|foreign body|without incision|tube)'
   )
)
'@

$MimicCohort = @'
, candidate_stays AS (
 SELECT i.subject_id,i.hadm_id,i.stay_id,i.first_careunit,i.intime,i.outtime,
        max(p.chartdate) AS latest_procedure_date,
        bool_or(p.chartdate=i.intime::date) AS has_same_day_code,
        bool_or(p.chartdate=i.intime::date-1) AS has_previous_day_code,
        string_agg(DISTINCT cm.surgery_family,';' ORDER BY cm.surgery_family) AS surgery_families,
        string_agg(DISTINCT (p.icd_version::text||':'||trim(p.icd_code)),';' ORDER BY (p.icd_version::text||':'||trim(p.icd_code))) AS qualifying_codes
 FROM mimiciv_icu.icustays i
 JOIN mimiciv_hosp.procedures_icd p ON p.hadm_id=i.hadm_id AND p.subject_id=i.subject_id
 JOIN code_map cm ON cm.icd_version=p.icd_version AND cm.icd_code=trim(p.icd_code)
 WHERE p.chartdate BETWEEN i.intime::date-1 AND i.intime::date
 GROUP BY i.subject_id,i.hadm_id,i.stay_id,i.first_careunit,i.intime,i.outtime
), evidence AS (
 SELECT c.*,p.anchor_age+(extract(year from c.intime)::int-p.anchor_year) AS age_at_icu,
        svc.curr_service AS service_at_icu,pacu.last_pacu_outtime,
        CASE
          WHEN pacu.last_pacu_outtime IS NOT NULL THEN 'A_PACU_WITHIN_24H'
          WHEN c.has_same_day_code AND svc.curr_service IN ('SURG','PSURG','GYN','GU','TRAUM','TSURG') THEN 'B_SURG_SERVICE_SAME_DATE'
          WHEN c.has_previous_day_code AND svc.curr_service IN ('SURG','PSURG','GYN','GU','TRAUM','TSURG') THEN 'C_SURG_SERVICE_PREVIOUS_DATE'
          ELSE 'UNSUPPORTED'
        END AS alignment_tier
 FROM candidate_stays c
 JOIN mimiciv_hosp.patients p ON p.subject_id=c.subject_id
 LEFT JOIN LATERAL (
   SELECT s.curr_service FROM mimiciv_hosp.services s
   WHERE s.hadm_id=c.hadm_id AND s.transfertime<=c.intime
   ORDER BY s.transfertime DESC LIMIT 1
 ) svc ON true
 LEFT JOIN LATERAL (
   SELECT max(t.outtime) AS last_pacu_outtime FROM mimiciv_hosp.transfers t
   WHERE t.hadm_id=c.hadm_id AND t.careunit='PACU'
     AND t.outtime<=c.intime AND t.outtime>=c.intime-INTERVAL '24 hour'
 ) pacu ON true
), adult AS (
 SELECT * FROM evidence WHERE age_at_icu>=18
), aligned AS (
 SELECT * FROM adult WHERE alignment_tier<>'UNSUPPORTED'
), first_eligible AS (
 SELECT * FROM (
   SELECT a.*,row_number() OVER (PARTITION BY subject_id ORDER BY intime,stay_id) AS rn
   FROM aligned a
 ) q WHERE rn=1
), landmark AS (
 SELECT f.*,a.deathtime,p.dod
 FROM first_eligible f
 JOIN mimiciv_hosp.admissions a ON a.hadm_id=f.hadm_id AND a.subject_id=f.subject_id
 JOIN mimiciv_hosp.patients p ON p.subject_id=f.subject_id
 WHERE f.outtime>=f.intime+INTERVAL '12 hour'
   AND (a.deathtime IS NULL OR a.deathtime>f.intime+INTERVAL '12 hour')
), outcome AS (
 SELECT l.*,
        CASE WHEN l.deathtime IS NULL AND l.dod IS NOT NULL
                       AND (l.dod=(l.intime+INTERVAL '12 hour')::date OR l.dod=(l.intime+INTERVAL '28 day')::date)
             THEN false ELSE true END AS outcome_constructible,
        CASE WHEN l.deathtime>l.intime+INTERVAL '12 hour' AND l.deathtime<=l.intime+INTERVAL '28 day' THEN 1
             WHEN l.deathtime IS NULL AND l.dod>(l.intime+INTERVAL '12 hour')::date AND l.dod<(l.intime+INTERVAL '28 day')::date THEN 1
             ELSE 0 END AS death28
 FROM landmark l
), final AS (
 SELECT * FROM outcome WHERE outcome_constructible
)
'@

$Queries = @(
 [pscustomobject]@{Database='mimiciv';Name='surgery_code_map';Sql=$MimicCodeMap+@'
SELECT icd_version,icd_code,long_title,surgery_family
FROM code_map
ORDER BY icd_version,icd_code
'@},
 [pscustomobject]@{Database='mimiciv';Name='flow_counts';Sql=$MimicCodeMap+$MimicCohort+@'
SELECT (SELECT count(*) FROM mimiciv_icu.icustays) AS raw_icu,
       (SELECT count(*) FROM mimiciv_icu.icustays i JOIN mimiciv_hosp.patients p USING(subject_id) WHERE p.anchor_age+(extract(year from i.intime)::int-p.anchor_year)>=18) AS adult,
       (SELECT count(*) FROM adult) AS major_abdominal_surgery_evidence,
       (SELECT count(*) FROM aligned) AS postoperative_aligned,
       (SELECT count(*) FROM first_eligible) AS first_eligible,
       (SELECT count(*) FROM landmark) AS landmark_alive_and_in_icu,
       (SELECT count(*) FROM final) AS outcome_constructible,
       (SELECT count(*) FROM final) AS final_n,
       (SELECT sum(death28) FROM final) AS death28_events
'@},
 [pscustomobject]@{Database='mimiciv';Name='alignment_final';Sql=$MimicCodeMap+$MimicCohort+@'
SELECT alignment_tier,count(*) AS adult_stays,
       round(100.0*count(*)/sum(count(*)) OVER (),2) AS percent_of_adult_surgery_evidence
FROM adult GROUP BY alignment_tier ORDER BY alignment_tier
'@},
 [pscustomobject]@{Database='mimiciv';Name='final_cohort_audit_sample50';Sql=$MimicCodeMap+$MimicCohort+@'
SELECT subject_id,hadm_id,stay_id,first_careunit,intime,outtime,age_at_icu,
       latest_procedure_date,surgery_families,qualifying_codes,service_at_icu,
       last_pacu_outtime,alignment_tier,outcome_constructible,death28
FROM final ORDER BY md5(stay_id::text||'20260918') LIMIT 50
'@},
 [pscustomobject]@{Database='sicdb';Name='surgery_site_map';Sql=@'
SELECT "ReferenceGlobalID" AS surgical_site_id,"ReferenceValue" AS surgical_site,
       CASE WHEN "ReferenceGlobalID" IN (2211,2219) THEN 'gastrointestinal'
            WHEN "ReferenceGlobalID" IN (2225,2242,2253) THEN 'hepatobiliary_pancreatic' END AS surgery_family
FROM public.d_references WHERE "ReferenceGlobalID" IN (2211,2219,2225,2242,2253)
ORDER BY "ReferenceGlobalID"
'@},
 [pscustomobject]@{Database='sicdb';Name='flow_counts';Sql=@'
WITH adult AS (
 SELECT * FROM public.cases WHERE "AgeOnAdmission">=18
), surgery AS (
 SELECT * FROM adult WHERE "SurgicalSite" IN (2211,2219,2225,2242,2253)
   AND "SurgicalAdmissionType" IN (3125,3126)
), aligned AS (
 SELECT * FROM surgery WHERE "ICUOffset">=0 AND "ICUOffset"<=86400
), first_eligible AS (
 SELECT * FROM (
  SELECT a.*,row_number() OVER (PARTITION BY "PatientID" ORDER BY coalesce("OffsetAfterFirstAdmission",0),"CaseID") AS rn
  FROM aligned a
 ) q WHERE rn=1
), landmark AS (
 SELECT * FROM first_eligible WHERE "TimeOfStay">="ICUOffset"+43200
   AND ("OffsetOfDeath" IS NULL OR "OffsetOfDeath">"ICUOffset"+43200)
), final AS (
 SELECT *,CASE WHEN "OffsetOfDeath">"ICUOffset"+43200 AND "OffsetOfDeath"<="ICUOffset"+28*86400 THEN 1 ELSE 0 END AS death28
 FROM landmark
)
SELECT (SELECT count(*) FROM public.cases) AS raw_icu,
       (SELECT count(*) FROM adult) AS adult,
       (SELECT count(*) FROM surgery) AS major_abdominal_surgery_evidence,
       (SELECT count(*) FROM aligned) AS postoperative_aligned,
       (SELECT count(*) FROM first_eligible) AS first_eligible,
       (SELECT count(*) FROM landmark) AS landmark_alive_and_in_icu,
       (SELECT count(*) FROM final) AS outcome_constructible,
       (SELECT count(*) FROM final) AS final_n,
       (SELECT sum(death28) FROM final) AS death28_events
'@},
 [pscustomobject]@{Database='sicdb';Name='alignment_final';Sql=@'
WITH surgery AS (
 SELECT *,CASE WHEN "ICUOffset">0 AND "ICUOffset"<=86400 THEN 'A_PRECEDING_SURGERY_CAPTURED_WITHIN_24H'
               WHEN "ICUOffset"=0 THEN 'B_DIRECT_SURGERY_ADMISSION_NO_PRE_ICU_INTERVAL'
               ELSE 'UNSUPPORTED_GT24H' END AS alignment_tier
 FROM public.cases WHERE "AgeOnAdmission">=18
   AND "SurgicalSite" IN (2211,2219,2225,2242,2253)
   AND "SurgicalAdmissionType" IN (3125,3126)
)
SELECT alignment_tier,count(*) AS cases,
       round(100.0*count(*)/sum(count(*)) OVER (),2) AS percent_of_adult_surgery_evidence
FROM surgery GROUP BY alignment_tier ORDER BY alignment_tier
'@},
 [pscustomobject]@{Database='sicdb';Name='final_cohort_audit_sample50';Sql=@'
WITH aligned AS (
 SELECT * FROM public.cases WHERE "AgeOnAdmission">=18
   AND "SurgicalSite" IN (2211,2219,2225,2242,2253)
   AND "SurgicalAdmissionType" IN (3125,3126)
   AND "ICUOffset">=0 AND "ICUOffset"<=86400
), first_eligible AS (
 SELECT * FROM (SELECT a.*,row_number() OVER (PARTITION BY "PatientID" ORDER BY coalesce("OffsetAfterFirstAdmission",0),"CaseID") AS rn FROM aligned a) q WHERE rn=1
), final AS (
 SELECT *,CASE WHEN "OffsetOfDeath">"ICUOffset"+43200 AND "OffsetOfDeath"<="ICUOffset"+28*86400 THEN 1 ELSE 0 END AS death28
 FROM first_eligible WHERE "TimeOfStay">="ICUOffset"+43200 AND ("OffsetOfDeath" IS NULL OR "OffsetOfDeath">"ICUOffset"+43200)
)
SELECT "CaseID","PatientID","AgeOnAdmission","AdmissionYear","SurgicalSite","SurgicalAdmissionType",
       "ICUOffset","TimeOfStay","OffsetOfDeath","EstimatedSurvivalObservationTime","OffsetAfterFirstAdmission",death28
FROM final ORDER BY md5("CaseID"::text||'20260918') LIMIT 50
'@}
)

$env:PGPASSWORD=$PlainPassword
$env:PGOPTIONS='-c default_transaction_read_only=on -c statement_timeout=600000 -c lock_timeout=5000'
$Run=@()
try {
 foreach($Query in $Queries){$Run+=Invoke-ReadOnlyCsv -Database $Query.Database -Name $Query.Name -Sql $Query.Sql}
 $Run | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Output 'run_summary.json') -Encoding utf8
 [pscustomobject]@{status='SUCCESS';output=$Output;artifacts=$Run.Count;completed_at=(Get-Date).ToString('o')} | ConvertTo-Json
} finally {
 Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
 Remove-Item Env:PGOPTIONS -ErrorAction SilentlyContinue
 $PlainPassword=$null
}

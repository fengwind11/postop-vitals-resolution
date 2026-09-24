[CmdletBinding()]
param(
    [SecureString]$PgPassword,
    [string]$PsqlPath = 'psql',
    [string]$PgHost = 'localhost',
    [string]$PgUser = 'postgres',
    [int]$PgPort = 5432
)

$ErrorActionPreference = 'Stop'
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$AuditRoot = Split-Path -Parent $ScriptRoot
$OutputRoot = Join-Path $AuditRoot '06_logs\internal_extracts'
$LogRoot = Join-Path $AuditRoot '06_logs'
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
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
    param([string]$Database,[string]$Name,[string]$Sql,[int]$TimeoutSeconds=1800)
    $Path = Join-Path $OutputRoot ("{0}.csv" -f $Name)
    if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Force }
    $psi = [Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $PsqlPath
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    foreach ($arg in @('-X','-w','--csv','-P','pager=off','-v','ON_ERROR_STOP=1','-h',$PgHost,'-p',([string]$PgPort),'-U',$PgUser,'-d',$Database,'-o',$Path,'-c',$Sql)) {
        [void]$psi.ArgumentList.Add($arg)
    }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $psi
    [void]$process.Start()
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        try { $process.Kill($true) } catch {}
        throw "psql timeout for $Database/$Name after $TimeoutSeconds seconds"
    }
    if ($process.ExitCode -ne 0) { throw "psql failed for $Database/$Name (exit $($process.ExitCode)): $stderr" }
    if (-not (Test-Path -LiteralPath $Path)) { throw "missing output for $Database/$Name" }
    [pscustomobject]@{
        database=$Database
        artifact=(Split-Path -Leaf $Path)
        bytes=(Get-Item -LiteralPath $Path).Length
        sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
        stderr=$stderr.Trim()
        stdout=$stdout.Trim()
    }
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

$SicdbCohort = @'
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
'@

$Queries = @(
    [pscustomobject]@{Database='mimiciv';Name='mimic_db_version';Sql="SELECT current_database() AS database, version() AS postgres_version, current_setting('server_version') AS server_version, now() AS extracted_at";Timeout=60},
    [pscustomobject]@{Database='mimiciv';Name='mimic_dictionary';Sql=@'
SELECT itemid,label,unitname,param_type,lownormalvalue,highnormalvalue
FROM mimiciv_icu.d_items
WHERE itemid IN (220045,220052,220181,220277,220210,224690,223762,223761)
ORDER BY itemid
'@;Timeout=60},
    [pscustomobject]@{Database='mimiciv';Name='mimic_cohort';Sql=$MimicCodeMap+$MimicCohort+@'
SELECT subject_id,hadm_id,stay_id,intime,outtime,first_careunit,alignment_tier
FROM final ORDER BY stay_id
'@;Timeout=600},
    [pscustomobject]@{Database='mimiciv';Name='mimic_signals_raw';Sql=$MimicCodeMap+$MimicCohort+@'
SELECT f.stay_id,f.intime,ce.itemid,ce.charttime,ce.valuenum,ce.valueuom
FROM final f
JOIN mimiciv_icu.chartevents ce ON ce.stay_id=f.stay_id
WHERE ce.itemid IN (220045,220052,220181,220277,220210,224690,223762,223761)
  AND ce.charttime>=f.intime
  AND ce.charttime<f.intime+INTERVAL '12 hour'
  AND ce.valuenum IS NOT NULL
ORDER BY f.stay_id,ce.charttime,ce.itemid
'@;Timeout=1800},
    [pscustomobject]@{Database='sicdb';Name='sicdb_db_version';Sql="SELECT current_database() AS database, version() AS postgres_version, current_setting('server_version') AS server_version, now() AS extracted_at";Timeout=60},
    [pscustomobject]@{Database='sicdb';Name='sicdb_dictionary';Sql=@'
SELECT "ReferenceGlobalID" AS dataid,"ReferenceValue" AS label,"ReferenceUnit" AS unit,
       "ReferenceName" AS reference_name
FROM public.d_references
WHERE "ReferenceGlobalID" IN (707,724,708,703,706,710,719,709)
ORDER BY "ReferenceGlobalID"
'@;Timeout=60},
    [pscustomobject]@{Database='sicdb';Name='sicdb_cohort';Sql=$SicdbCohort+@'
SELECT "CaseID" AS caseid,"PatientID" AS patientid,"ICUOffset" AS icuoffset,
       "TimeOfStay" AS timeofstay,"OffsetAfterFirstAdmission" AS offsetafterfirstadmission
FROM final ORDER BY "CaseID"
'@;Timeout=600},
    [pscustomobject]@{Database='sicdb';Name='sicdb_signals_hourly_raw';Sql=$SicdbCohort+@'
SELECT f."CaseID" AS caseid,f."ICUOffset" AS icuoffset,d."DataID" AS dataid,
       d."Offset" AS row_offset,d."Val" AS stored_val,d.cnt,d.rawdata
FROM final f
JOIN public.data_float_h d ON d."CaseID"=f."CaseID"
WHERE d."DataID" IN (707,724,708,703,706,710,719,709)
  AND d."Offset">=f."ICUOffset"-3600
  AND d."Offset"<f."ICUOffset"+43200
ORDER BY f."CaseID",d."DataID",d."Offset"
'@;Timeout=1800}
)

$env:PGPASSWORD = $PlainPassword
$env:PGOPTIONS = '-c default_transaction_read_only=on -c statement_timeout=1800000 -c lock_timeout=5000'
$Run = @()
try {
    foreach ($Query in $Queries) {
        $started = Get-Date
        $row = Invoke-ReadOnlyCsv -Database $Query.Database -Name $Query.Name -Sql $Query.Sql -TimeoutSeconds $Query.Timeout
        $row | Add-Member -NotePropertyName started_at -NotePropertyValue $started.ToString('o')
        $row | Add-Member -NotePropertyName completed_at -NotePropertyValue (Get-Date).ToString('o')
        $Run += $row
    }
    $MimicN = (Import-Csv -LiteralPath (Join-Path $OutputRoot 'mimic_cohort.csv')).Count
    $SicdbN = (Import-Csv -LiteralPath (Join-Path $OutputRoot 'sicdb_cohort.csv')).Count
    if ($MimicN -ne 1740 -or $SicdbN -ne 2376) {
        throw "STOP_AND_DEBUG_COHORT: recovered MIMIC=$MimicN (expected 1740), SICdb=$SicdbN (expected 2376)"
    }
    $Scripts = Get-ChildItem -LiteralPath $ScriptRoot -File | ForEach-Object {
        [pscustomobject]@{file=$_.Name;sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()}
    }
    $Manifest = [pscustomobject]@{
        status='SUCCESS'
        seed=20260918
        mimic_n=$MimicN
        sicdb_n=$SicdbN
        read_only=$true
        completed_at=(Get-Date).ToString('o')
        files=$Run
        scripts=$Scripts
    }
    $Manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $LogRoot 'extraction_manifest.json') -Encoding utf8
    $Manifest | ConvertTo-Json -Depth 4
} finally {
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    Remove-Item Env:PGOPTIONS -ErrorAction SilentlyContinue
    $PlainPassword = $null
}

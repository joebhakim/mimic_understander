, long_dx_table as

(
select diagnoses_icd.hadm_id
, lower(ccs_name) like '%essential hypertension%' as hypertension
, lower(ccs_name) like  "Coronary atherosclerosis"


, lower(ccs_name) like "%disorders of lipid metabolism"
, lower(ccs_name) like "%congestive heart failure"
, lower(ccs_name) like "%atrial fibrillation"
, lower(ccs_name) like "%acute renal failure"
, lower(ccs_name) like "%unspecified septicemia"
, lower(ccs_name) like "%diabetes mellitus without complication "
, lower(ccs_name) like "%other fluid and electrolyte disorders"
, lower(ccs_name) like "%other aftercare"
, lower(ccs_name) like "%respiratory failure"
, lower(ccs_name) like "%chronic kidney disease "
, lower(ccs_name) like "%other esophageal disorders"
, lower(ccs_name) like "%congestive heart failure; nonhypertensive "
, lower(ccs_name) like "%other forms of chronic heart disease"
, lower(ccs_name) like "%hypertensive heart and/or renal disease"
, lower(ccs_name) like "%alcohol-related disorders"

, lower(ccs_name) like '%meningitis%' as meningitis
, lower(ccs_name) like '%pneumonia%' as pneumonia

from diagnoses_icd
inner join ccs_dx on ccs_dx.icd9_code = diagnoses_icd.icd9_code
where diagnoses_icd.seq_num <= 3

)
, pivoted_dx_table as 
(
select hadm_id
, bool_or(meningitis) as meningitis
, bool_or(pneumonia) as pneumonia
from long_dx_table
group by hadm_id
)

select distinct ep.subject_id, ep.hadm_id, ep.icustay_id
    , i.age
    , eq.elixhauser_sid30 as elixhauser
    , sofa.sofa
    , gcsfirstday.mingcs as gcsfirstday_mingcs
    , elixhauser_quan.congestive_heart_failure
    , dx.meningitis
    , dx.pneumonia
    , i.gender
    , i.ethnicity_grouped
    
    FROM eligible_patients ep
    LEFT JOIN icustay_detail i on i.icustay_id = ep.icustay_id
    LEFT JOIN admissions a ON ep.hadm_id = a.hadm_id
    LEFT JOIN sofa on sofa.icustay_id = ep.icustay_id
    LEFT JOIN elixhauser_quan_score eq on eq.hadm_id = ep.hadm_id
    LEFT JOIN elixhauser_quan on elixhauser_quan.hadm_id = ep.hadm_id
    LEFT JOIN gcsfirstday on gcsfirstday.icustay_id = ep.icustay_id
    left join diagnoses_icd on diagnoses_icd.hadm_id = ep.hadm_id
    left join ccs_dx on ccs_dx.icd9_code = diagnoses_icd.icd9_code
    left join pivoted_dx_table dx on dx.hadm_id = ep.hadm_id


select distinct ep.subject_id, ep.hadm_id, ep.stay_id
, g.age
, a.ethnicity
FROM eligible_patients ep


LEFT JOIN `physionet-data.mimic_derived.age` g on g.hadm_id = ep.hadm_id
LEFT JOIN `physionet-data.mimic_core.admissions` a ON a.hadm_id = ep.hadm_id
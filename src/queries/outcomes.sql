select distinct ep.subject_id, ep.hadm_id, ep.stay_id,
    --CASE when a.deathtime between i.intime and i.outtime THEN 1 ELSE 0 END AS mort_icu,
    --CASE when a.deathtime between i.admittime and i.dischtime THEN 1 ELSE 0 END AS mort_hosp,
    a.hospital_expire_flag
    --i.los_icu
    , CASE WHEN a.deathtime <= DATETIME_ADD(a.admittime, interval '30' day) THEN 1 ELSE 0 END AS HospMort30day -- % 30 day mortality
    FROM eligible_patients ep 
    --LEFT JOIN icustay_detail i on i.icustay_id = ep.icustay_id
    LEFT JOIN `physionet-data.mimic_core.admissions` a ON ep.hadm_id = a.hadm_id
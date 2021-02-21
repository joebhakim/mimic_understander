with eligible_patients as
(
select distinct i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime
	FROM `physionet-data.mimic_icu.icustays` i
	INNER JOIN `physionet-data.mimic_core.admissions` a ON i.hadm_id = a.hadm_id
	INNER JOIN `physionet-data.mimic_derived.age` g ON i.hadm_id = i.hadm_id
	where
		i.subject_id >= 40000
		and g.age >= 18
		and i.los >= 0.5
		and i.outtime >= DATETIME_ADD(i.intime, INTERVAL 12 HOUR)
		and i.outtime <= DATETIME_ADD(i.intime, INTERVAL 240 HOUR)
)
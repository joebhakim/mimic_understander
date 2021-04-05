with eligible_patients as
(
select distinct i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime
	FROM `physionet-data.mimic_icu.icustays` i
	--INNER JOIN `physionet-data.mimic_core.admissions` admissions ON i.hadm_id = admissions.hadm_id
	INNER JOIN `physionet-data.mimic_derived.age` age ON i.hadm_id = age.hadm_id
	inner join `physionet-data.mimic_hosp.diagnoses_icd` dx on dx.hadm_id = i.hadm_id
	where
		i.subject_id >= 40000
		and age.age >= 18
		and i.los >= 0.5
		and i.outtime >= DATETIME_ADD(i.intime, INTERVAL 12 HOUR)
		and i.outtime <= DATETIME_ADD(i.intime, INTERVAL 240 HOUR)
		and icd_code like 'I48%' or icd_code like '42731%' --atrial fibrillation or atrial flutter
)


select distinct ep.subject_id, ep.hadm_id, ep.stay_id, ep.intime, ep.outtime
, c.charttime, id.itemid, id.label, c.value, valueuom
	FROM eligible_patients ep
    LEFT JOIN `physionet-data.mimic_icu.chartevents` c ON ep.stay_id = c.stay_id
    LEFT JOIN `physionet-data.mimic_icu.d_items` id on id.itemid = c.itemid
    where c.charttime between intime and outtime
            and c.warning is distinct from 1
            and c.valuenum is not null
limit 100000
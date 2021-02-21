select distinct ep.subject_id, ep.hadm_id, ep.icustay_id, ep.intime, ep.outtime,
        c.charttime, id.itemid, id.label, c.value, valueuom
    FROM eligible_patients ep
    LEFT JOIN chartevents c ON ep.icustay_id = c.icustay_id
    LEFT JOIN d_items id on id.itemid = c.itemid
    where c.charttime between intime and outtime
        and c.error is distinct from 1
        and c.valuenum is not null
    UNION ALL
    select distinct ep.subject_id, ep.hadm_id, ep.icustay_id, ep.intime, ep.outtime,
        l.charttime, id.itemid, id.label, l.value, valueuom
    FROM eligible_patients ep
    LEFT JOIN labevents l ON ep.hadm_id = l.hadm_id
    LEFT JOIN d_labitems id on id.itemid = l.itemid
    --where l.charttime between (intime - interval '6' hour) and outtime
     --   and l.valuenum > 0 -- lab values cannot be 0 and cannot be negative
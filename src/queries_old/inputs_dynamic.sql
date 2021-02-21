
select distinct ep.subject_id, ep.hadm_id, ep.icustay_id
    , ep.intime, ep.outtime
    , inputevents_mv.starttime, inputevents_mv.endtime
    , inputevents_mv.amount, inputevents_mv.amountuom
    , inputevents_mv.rate, inputevents_mv.rateuom
    , d_items.label, d_items.itemid
from eligible_patients ep
LEFT JOIN inputevents_mv ON ep.icustay_id = inputevents_mv.icustay_id
LEFT join d_items on d_items.itemid = inputevents_mv.itemid
where cancelreason = 0

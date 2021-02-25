select distinct ep.subject_id, ep.hadm_id, ep.stay_id
    , ep.intime, ep.outtime
    , inputevents.starttime, inputevents.endtime
    , inputevents.amount, inputevents.amountuom
    , inputevents.rate, inputevents.rateuom
    , d_items.label, d_items.itemid
from eligible_patients ep
LEFT JOIN `physionet-data.mimic_icu.inputevents` inputevents ON ep.stay_id = inputevents.stay_id
LEFT join `physionet-data.mimic_icu.d_items` d_items on d_items.itemid = inputevents.itemid
where cancelreason = 0
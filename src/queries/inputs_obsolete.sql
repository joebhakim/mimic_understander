, full_inputs6hrs as 
(
    select inputevents_mv.subject_id, inputevents_mv.hadm_id, inputevents_mv.icustay_id
    , inputevents_mv.starttime, inputevents_mv.endtime
    , inputevents_mv.amount, inputevents_mv.amountuom
    , d_items.label, d_items.itemid
    , icustay_detail.admittime
    from inputevents_mv 
    inner join d_items on d_items.itemid = inputevents_mv.itemid
    inner join icustay_detail on icustay_detail.icustay_id = inputevents_mv.icustay_id
    --where starttime < icustay_detail.admittime + interval '6' hour
    and cancelreason = 0
)
, frequent_drugs as 
(
    select label, itemid, count(label) as count_label
    from full_inputs6hrs
    group by label, itemid
    order by count_label desc
    limit 15
)
, pivoted as 
(
    select full_inputs6hrs.subject_id, full_inputs6hrs.hadm_id, full_inputs6hrs.icustay_id
    , case when full_inputs6hrs.label = 'NaCl 0.9%' then sum(amount) else 0 end as sum_NaCl
    , case when full_inputs6hrs.label = 'Dextrose 5%' then sum(amount) else 0 end as sum_Dextrose
    , case when full_inputs6hrs.label = 'Solution' then sum(amount) else 0 end as sum_Solution
    , case when full_inputs6hrs.label = 'Propofol' then sum(amount) else 0 end as sum_Propofol
    , case when full_inputs6hrs.label = 'Pre-Admission Intake' then sum(amount) else 0 end as sum_PreAdmission
    , case when full_inputs6hrs.label = 'Norepinephrine' then sum(amount) else 0 end as sum_Norepinephrine
    , case when full_inputs6hrs.label = 'Midazolam (Versed)' then sum(amount) else 0 end as sum_Midazolam
    , case when full_inputs6hrs.label = 'Phenylephrine' then sum(amount) else 0 end as sum_Phenylephrine
    , case when full_inputs6hrs.label = 'LR' then sum(amount) else 0 end as sum_LR
    , case when full_inputs6hrs.label = 'PO Intake' then sum(amount) else 0 end as sum_PO
    , case when full_inputs6hrs.label = 'Fentanyl' then sum(amount) else 0 end as sum_Fentanyl
    , case when full_inputs6hrs.label = 'Insulin - Regular' then sum(amount) else 0 end as sum_Insulin
    , case when full_inputs6hrs.label = 'Potassium Chloride' then sum(amount) else 0 end as sum_Potassium
    , case when full_inputs6hrs.label = 'Packed Red Blood Cells' then sum(amount) else 0 end as sum_PackedRBC
    , case when full_inputs6hrs.label = 'Nitroglycerin' then sum(amount) else 0 end as sum_Nitroglycerin
    from full_inputs6hrs
    inner join frequent_drugs on full_inputs6hrs.itemid = frequent_drugs.itemid 

    GROUP BY full_inputs6hrs.subject_id, full_inputs6hrs.hadm_id, full_inputs6hrs.icustay_id, full_inputs6hrs.label
    ORDER BY full_inputs6hrs.subject_id, full_inputs6hrs.hadm_id, full_inputs6hrs.icustay_id
)
select distinct ep.subject_id, ep.hadm_id, ep.icustay_id
, sum(pivoted.sum_NaCl) as sum_NaCl
, sum(pivoted.sum_Dextrose) as sum_Dextrose
, sum(pivoted.sum_Solution) as sum_Solution
, sum(pivoted.sum_Propofol) as sum_Propofol
, sum(pivoted.sum_PreAdmission) as sum_PreAdmission
, sum(pivoted.sum_Norepinephrine) as sum_Norepinephrine
, sum(pivoted.sum_Midazolam) as sum_Midazolam
, sum(pivoted.sum_Phenylephrine) as sum_Phenylephrine
, sum(pivoted.sum_LR) as sum_LR
, sum(pivoted.sum_PO) as sum_PO
, sum(pivoted.sum_Fentanyl) as sum_Fentanyl
, sum(pivoted.sum_Insulin) as sum_Insulin
, sum(pivoted.sum_Potassium) as sum_Potassium
, sum(pivoted.sum_PackedRBC) as sum_PackedRBC
, sum(pivoted.sum_Nitroglycerin) as sum_Nitroglycerin
from eligible_patients ep
left join pivoted on pivoted.icustay_id = ep.icustay_id
group by ep.subject_id, ep.hadm_id, ep.icustay_id
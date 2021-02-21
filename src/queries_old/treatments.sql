SELECT distinct ep.subject_id, ep.hadm_id, ep.icustay_id,
    (
        (LOWER(DRUG) LIKE '%hydrocort%' 
        OR LOWER(DRUG_NAME_POE) LIKE '%hydrocort%' 
        OR LOWER(DRUG_NAME_GENERIC) LIKE '%hydrocort%'
        OR LOWER(DRUG) LIKE '%dexamethasone%' 
        OR LOWER(DRUG_NAME_POE) LIKE '%dexamethasone%' 
        OR LOWER(DRUG_NAME_GENERIC) LIKE '%dexamethasone%'
        OR LOWER(DRUG) LIKE '%methylprednisolone%' 
        OR LOWER(DRUG_NAME_POE) LIKE '%methylprednisolone%' 
        OR LOWER(DRUG_NAME_GENERIC) LIKE '%methylprednisolone%')
        AND 
        (DRUG != 'Hydrocortisone Study Drug (*IND*)' 
        OR DRUG_NAME_POE != 'Hydrocortisone Study Drug (*IND*)' 
        OR DRUG_NAME_GENERIC != 'Hydrocortisone Study Drug (*IND*)')
        AND 
        ROUTE = 'IV'
        AND 
        startdate > icustay_detail.admittime + interval '6' hour
    ) as is_on_cort
    FROM eligible_patients ep
    LEFT JOIN prescriptions on prescriptions.icustay_id = ep.icustay_id
    LEFT JOIN icustay_detail on icustay_detail.icustay_id = ep.icustay_id
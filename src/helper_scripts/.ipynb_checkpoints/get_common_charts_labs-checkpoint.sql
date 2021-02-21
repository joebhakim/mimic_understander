set search_path to mimiciii;

with sample_charts as
(
	SELECT 	
		ce.itemid,
		ce.valueuom,
		'chart' as chartorlab,
		di.label
	FROM chartevents ce
	inner join d_items di on di.itemid = ce.itemid
	where ce.subject_id >= 40000
	LIMIT 100000
	
),
sample_labs as 
(
	select 
		le.itemid,
		le.valueuom,
		'lab' as chartorlab,
		di.label
	FROM labevents le
	inner join d_labitems di on di.itemid = le.itemid
	where le.subject_id >= 40000
	LIMIT 100000
), 
sample as
(
	select *
	FROM sample_charts ce
	union all 
	select *
	from sample_labs le
)

SELECT count(sample.itemid), sample.itemid,
sample.label, sample.valueuom, sample.chartorlab
FROM sample
group by sample.itemid, sample.label, sample.valueuom, sample.chartorlab
order by count(sample.itemid) desc
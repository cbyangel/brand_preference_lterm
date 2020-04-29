with a as
( select brd_id, count(*) as cnt from dealteminfoextra group by brd_id),
b as
( select distinct brand_cd, brand_nm from brand_info where brand_nm is not null and brand_cd is not null  )
select
    a.brd_id as brand,
    b.brand_nm as brd_nm,
    a.cnt
from a left join b
on a.brd_id = b.brand_cd
where
  b.brand_cd is not null

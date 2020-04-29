with sub1 as
(
    select
        t1.pcid,
        t2.brd_id,
        t2.cate1,
        t1.view_cnt
    from
        members_will.daily_prd_view_cnt t1
    left join
        gsshop.product t2
    on
        t1.prd_id = t2.prd_id
    where
        t2.prd_id is not null
)
select
    brd_id,
    cate1,
    count(distinct pcid) as view_cnt
from
    sub1
where
    brd_id is not null
and
    cate1 is not null
group by
    brd_id,
    cate1
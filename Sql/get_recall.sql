 with sub1 as
(
   select
        t1.pcid,
        t1.brd_id,
        sum(t1.view_cnt) as view_cnt,
        sum(t1.cart_cnt) + sum(t1.ord_cnt) as positive_cnt
    from
        members_bycho.daily_cust_brand t1
    left join
        members_bycho.brand_preference t2
    on
        t1.pcid = t2.userid
    where
        t2.userid is not null
    group by
        t1.pcid,
        t1.brd_id
),
sub2 as
(
    SELECT
        pcid,
        brd_id,
        round(percent_rank() OVER (PARTITION BY pcid order by view_cnt asc), 2) as n_view,
        round(percent_rank() OVER (PARTITION BY pcid order by positive_cnt asc), 2) as n_positive
    FROM sub1
),
sub3 as
(
    SELECT pcid, brd_id, n_view, n_positive,
        case when
            isnan(round( (n_view * n_positive) / ((n_view * n_positive) + (1 - n_view) * ( 1 - n_positive)), 2))
        then 0
        else round( (n_view * n_positive) / ((n_view * n_positive) + (1 - n_view) * ( 1 - n_positive)), 2) end as isprefer
    FROM sub2

),
sub4 as
(
    select
        pcid as userid,
        brd_id as brand,
        n_view,
        n_positive,
        isprefer,
        row_number() over(partition by pcid order by isprefer desc, n_view desc ) as rnk
    from
        sub3
)
select
    userid,
    brand,
    isprefer as score,
    rnk
from
    sub4
where
    rnk < 11

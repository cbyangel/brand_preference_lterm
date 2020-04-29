select
    s1.pcid,
    s1.sect_b_cd as cate1,
    sum(s1.view_cnt) as view_cnt
from
    members_will.daily_sect_view_cnt s1
left join
    members_bycho.brand_preference s2
on
    s1.pcid = s2.userid
where
    s2.userid is not null
group by
    s1.pcid,
    s1.sect_b_cd

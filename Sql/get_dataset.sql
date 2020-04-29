select
    t1.pcid,
    t1.brand,
    t1.is_prefer as istarget
from
    members_bycho.brand_target t1
left join
    members_bycho.brand_preference t2
on
    t1.pcid = t2.userid
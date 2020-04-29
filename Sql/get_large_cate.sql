select
  distinct sect_b_cd as cate1
from
  members_will.daily_sect_view_cnt
where
  sect_b_cd is not null
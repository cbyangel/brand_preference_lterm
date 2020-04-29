select
  userid,
  e_brand as brand
from
  members_bycho.brand_preference2
lateral view posexplode(split(brand, ',')) e_brand_idx AS seqb, e_brand



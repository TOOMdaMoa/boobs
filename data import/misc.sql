SELECT
  product_id,
  count(*) AS p
FROM user_views
GROUP BY product_id
ORDER BY p DESC;

-- pohlavi [muz, zena, kluk, holka]
-- variant_data_params_DETERMINATION, variant_data_params_GIRLS_SHOES, variant_data_params_MENS_SHOES, variant_data_params_MEN_WOMEN, variant_data_params_WOMENS_SHOES, variant_data_params_BOYS_SHOES

-- ucel [CASUAL, OUTDOOR, SPORT, DRESS]
-- variant_data_params_DETERMINATION_OF_SHOES

-- typ [sport, WINTER BOOTS, trek, letni, kazdodenni, lodicky, holinky, pantofle, baleriny]
-- variant_data_params_RUNNER, variant_data_params_SPORT, variant_data_params_TYPE_OF_SHOES

-- sport=SPORT SHOES, RUNNING SHOES, HALL SHOES,FITNESS SHOES, SPORTING
-- winter=WINTER BOOTS, ANKLE BOOTS, SNOWBOOTS, ANKLE WINTER, TURISTIC SHOES, FELT BOOTS, BOOTS
-- trek=TREKING SHOES, BACKPACKING BOOTS, LOW TOURISTIC, BACKPACKING BOOTS, LOW TOURISTIC
-- letni=CROCS, SANDALS, FLIP-FLOPS, ESPADRILLE
-- kazdodenni=TENNIS SHOES, SHOES, MOCCASINS
-- lodicky=HEELS
-- holinky=GUM BOOT
-- pantofle=SLIPPERS, HOME
-- baleriny=BALLERINAS

-- cena - kdyz se bude fakt brutalne lisit
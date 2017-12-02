DROP TABLE IF EXISTS user_views;


CREATE TABLE user_views
(
  id                                                 BIGSERIAL PRIMARY KEY NOT NULL UNIQUE,
  brand_brand_id                                     TEXT,
  brand_title_c                                      TEXT,
  brief_plain_c                                      TEXT,
  date                                               TIMESTAMP WITH TIME ZONE,
  index_missing                                      BOOLEAN,
  index_out_of_range                                 BOOLEAN,
  product_id                                         BIGINT                NOT NULL,
  product_type_id                                    TEXT,
  product_type_title_c                               TEXT,
  title_full_c                                       TEXT,
  user_id                                            TEXT                  NOT NULL,
  variant_data_availability_cz_clearance_outlets     TEXT,
  variant_data_availability_cz_clearance_outlets_0   TEXT,
  variant_data_availability_cz_clearance_outlets_1   TEXT,
  variant_data_availability_cz_external_availability TEXT,
  variant_data_availability_cz_external_term_days    TEXT,
  variant_data_availability_cz_is_available          TEXT,
  variant_data_availability_cz_numeric               TEXT,
  variant_data_availability_cz_our_availability      TEXT,
  variant_data_availability_cz_outlets               TEXT,
  variant_data_availability_cz_outlets_1             TEXT,
  variant_data_availability_cz_outlets_2             TEXT,
  variant_data_availability_cz_outlets_3             TEXT,
  variant_data_availability_cz_outlets_4             TEXT,
  variant_data_availability_cz_outlets_5             TEXT,
  variant_data_availability_cz_outlets_6             TEXT,
  variant_data_availability_cz_outlets_7             TEXT,
  variant_data_availability_cz_outlets_8             TEXT,
  variant_data_availability_cz_outlets_9             TEXT,
  variant_data_availability_cz_status                TEXT,
  variant_data_availability_cz_supplier_availability TEXT,
  variant_data_availability_cz_supplier_term_days    TEXT,
  variant_data_labels_cz10ma_0_label_id              TEXT,
  variant_data_labels_cz10ma_10_label_id             TEXT,
  variant_data_labels_cz10ma_11_label_id             TEXT,
  variant_data_labels_cz10ma_12_label_id             TEXT,
  variant_data_labels_cz10ma_13_label_id             TEXT,
  variant_data_labels_cz10ma_1_label_id              TEXT,
  variant_data_labels_cz10ma_2_label_id              TEXT,
  variant_data_labels_cz10ma_3_label_id              TEXT,
  variant_data_labels_cz10ma_4_label_id              TEXT,
  variant_data_labels_cz10ma_5_label_id              TEXT,
  variant_data_labels_cz10ma_6_label_id              TEXT,
  variant_data_labels_cz10ma_7_label_id              TEXT,
  variant_data_labels_cz10ma_8_label_id              TEXT,
  variant_data_labels_cz10ma_9_label_id              TEXT,
  variant_data_params_autumn_shoes                   TEXT,
  variant_data_params_boys_shoes                     TEXT,
  variant_data_params_collection_nm                  TEXT,
  variant_data_params_color                          TEXT,
  variant_data_params_determination                  TEXT,
  variant_data_params_determination_of_shoes         TEXT,
  variant_data_params_extended_selection             TEXT,
  variant_data_params_fashion                        TEXT,
  variant_data_params_girls_shoes                    TEXT,
  variant_data_params_heel                           TEXT,
  variant_data_params_material                       TEXT,
  variant_data_params_membrane                       TEXT,
  variant_data_params_mens_shoes                     TEXT,
  variant_data_params_men_women                      TEXT,
  variant_data_params_model_of_year                  TEXT,
  variant_data_params_runner                         TEXT,
  variant_data_params_season_of_year                 TEXT,
  variant_data_params_shoes_high                     TEXT,
  variant_data_params_size                           TEXT,
  variant_data_params_size_eur                       TEXT,
  variant_data_params_size_uk                        TEXT,
  variant_data_params_sport                          TEXT,
  variant_data_params_spring_shoes                   TEXT,
  variant_data_params_summer_shoes                   TEXT,
  variant_data_params_tie                            TEXT,
  variant_data_params_type_of_binding                TEXT,
  variant_data_params_type_of_shoes                  TEXT,
  variant_data_params_type_of_toe                    TEXT,
  variant_data_params_type_usage                     TEXT,
  variant_data_params_weight_g                       TEXT,
  variant_data_params_winter_shoes                   TEXT,
  variant_data_params_womens_shoes                   TEXT,
  variant_data_price_cz1000_currency                 TEXT,
  variant_data_price_cz1000_current_price            TEXT,
  variant_data_price_cz1000_price                    INT,
  variant_data_price_cz1000_promotion_amount         TEXT,
  variant_data_price_cz1000_promotion_end            TEXT,
  variant_data_price_cz1000_promotion_price          TEXT,
  variant_data_price_cz1000_promotion_start          TEXT,
  variant_data_price_cz1000_rrp                      TEXT,
  variant_data_price_cz1000_vat_rate                 TEXT,
  variant_data_title_5                               TEXT,
  variant_data_title_6                               TEXT,
  variant_data_title_c                               TEXT,
  variant_data_title_h                               TEXT,
  variant_data_title_l                               TEXT,
  variant_data_title_q                               TEXT,
  variant_id                                         TEXT
);


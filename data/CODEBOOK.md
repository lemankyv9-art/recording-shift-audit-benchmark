# Data dictionary

The primary `source.csv`, `target.csv` and `external_test.csv` contain 20 recorded factor flags. A one means the factor was recorded. A zero can mean absence or non-recording. `model_factors.csv` and `assessment_mask.csv` are supplied for diagnostics.

| Code | Meaning |
|---|---|
| H01 | speeding |
| H02 | fatigue |
| H05 | failure to yield |
| H08 | improper operation |
| H10 | unsafe following distance |
| H11 | unsafe lane change |
| V01 | vehicle modification |
| V02 | brake failure |
| V03 | steering failure |
| V04 | tire failure |
| V05 | lighting deficiency |
| V06 | safety-protection deficiency |
| E01 | adverse weather |
| E02 | roadwork |
| E04 | curved ramp |
| E05 | intersection |
| E06 | wet road |
| E07 | marking deficiency |
| E08 | road-design deficiency |
| E11 | ramp |

`E04=1` implies `E11=1`.

## Outcomes

`fatalities` is an integer count and `economic loss` is in million CNY under the declared model. `severity` is calculated last: major if fatalities >=10 or loss >=50; high if otherwise fatalities >=3 or loss >=10; medium if otherwise fatalities >=1 or loss >=1; low otherwise. It is a study-defined threshold label, not an official accident grade.

## Metadata and files

`metadata.csv` contains generated keys, source/target membership, split membership and row order. It contains no dates or report identifiers. Source has 343 rows; target has 121 rows, with 36 development and 85 holdout rows. `external_test.csv` has 2,000 rows and is not used for selection.

The `datasets/` tree contains all 30 replicates and three scenarios. `event_context.csv` contains mechanism diagnostics and loss components. `results/` contains fixed predictions, metrics, summaries and verification.

## Context diagnostics

All rows of event_context.csv align with source rows followed by all target rows; external_context.csv aligns with external_test.csv. No joins by a real report identifier are possible.

- context: one of truck_vehicle, truck_vru, truck_bus, tanker_community; region_group: hypothetical stratum index 0–3.
- trucks, cars, bus_involved, vehicles_involved: event vehicle counts. bus_seats and bus_occupants are zero without a bus. Occupancy includes drivers.
- car_occupants, vehicle_occupants: role-specific and total people inside vehicles.
- vulnerable_road_users, community_present, external_exposed, people_exposed: person counts. Only external_exposed enters total exposure, after the stated physical hazard pathway.
- hazardous_cargo, built_environment, fire, rollover, hazardous_release, crowd_collision: binary event indicators.
- occupant_fatalities, vru_fatalities, external_fatalities: nonoverlapping fatality counts whose sum equals fatalities.
- serious_injuries, minor_injuries, unhurt: nonoverlapping counts among those not fatally injured; with fatalities they sum to people_exposed.
- emergency_access and report_quality: model scores between zero and one. investigation_burden: model event-complexity score, not an observed official assessment.
- asset_value_million_cny: exposed assets; property_loss_million_cny, casualty_cost_million_cny, response_cost_million_cny: distinct model expense components.
- loss_raw_million_cny: sum before mixed rounding; loss_precision: rounding step (0.001, 0.01 or 0.1 million CNY). A reported zero may be a rounded small positive loss. Component storage has four decimals, raw loss eight, and reported loss up to three.

The convenience matrices are exact copies of r00 composition-shift files. Identical factor or 23-field vectors can arise from independent draws and rounding; keys distinguish computational draws. The source sample and certain target covariates are intentionally paired across scenarios.

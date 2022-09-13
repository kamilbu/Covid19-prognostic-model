CREATE TABLE poland_covid as
(WITH covid_data_processing as (select date_reported, country_code, country, new_cases, cumulative_cases, new_deaths, cumulative_deaths, 

case when date_reported < '2020-12-28' then 0 
else total_vaccinations end as 
total_vaccinations,
case when date_reported < '2020-12-28' then 0 
else people_vaccinated end as people_vaccinated, 
case when date_reported < '2021-01-17' then 0 
else people_fully_vaccinated end as people_fully_vaccinated, 
case when date_reported < '2021-10-25' then 0
else total_boosters end as total_boosters, 
case when date_reported < '2020-12-28' then 0 
when date_reported='2020-12-28' then total_vaccinations
else daily_vaccinations end as daily_vaccinations, 
case when date_reported < '2020-12-28' then 0 
else total_vaccinations_per_hundred end as total_vaccinations_per_hundred, 
case when date_reported < '2021-01-16' then 0 
else people_vaccinated_per_hundred end as people_vaccinated_per_hundred, 
case when date_reported < '2021-01-16' then 0 
else people_fully_vaccinated_per_hundred end as people_fully_vaccinated_per_hundred, 
case when date_reported < '2021-10-25' then 0 
else total_boosters_per_hundred end as total_boosters_per_hundred, 
case when date_reported < '2020-12-28' then 0 
else daily_vaccinations_per_million end as daily_vaccinations_per_million, 
case when date_reported < '2020-12-28' then 0 
when date_reported='2020-12-28' then people_vaccinated 
else daily_people_vaccinated end as daily_people_vaccinated,
case when date_reported < '2020-12-28' then 0 
else daily_people_vaccinated_per_hundred end as daily_people_vaccinated_per_hundred


from (select cases.*, vacc.* from
(select * from covid_global where country='Poland') cases
left join 
(select * from vaccination1 where location='Poland') vacc
on cases.date_reported=vacc.date)
where date_reported >= '2020-03-07')

select 
DATE_REPORTED
,COUNTRY_CODE
,COUNTRY
,NEW_CASES
,CUMULATIVE_CASES                          --ifnull(,previousnonnull_tv)
,NEW_DEATHS
,CUMULATIVE_DEATHS
,ifnull(total_vaccinations,previousnonnull_tv)TOTAL_VACCINATIONS
,ifnull( PEOPLE_VACCINATED,previousnonnull_pv) PEOPLE_VACCINATED
,ifnull(PEOPLE_FULLY_VACCINATED,previousnonnull_pfv) PEOPLE_FULLY_VACCINATED
,ifnull(TOTAL_BOOSTERS,previousnonnull_tb) TOTAL_BOOSTERS
,ifnull(DAILY_VACCINATIONS,previousnonnull_dv) DAILY_VACCINATIONS
,ifnull(TOTAL_VACCINATIONS_PER_HUNDRED,previousnonnull_tvph) TOTAL_VACCINATIONS_PER_HUNDRED
,ifnull(PEOPLE_VACCINATED_PER_HUNDRED,previousnonnull_pvph)PEOPLE_VACCINATED_PER_HUNDRED
,ifnull(PEOPLE_FULLY_VACCINATED_PER_HUNDRED,previousnonnull_pfvph) PEOPLE_FULLY_VACCINATED_PER_HUNDRED
,ifnull(TOTAL_BOOSTERS_PER_HUNDRED,previousnonnull_tbph) TOTAL_BOOSTERS_PER_HUNDRED
,ifnull(DAILY_VACCINATIONS_PER_MILLION,previousnonnull_dvpm) DAILY_VACCINATIONS_PER_MILLION
,ifnull(DAILY_PEOPLE_VACCINATED,previousnonnull_pv) DAILY_PEOPLE_VACCINATED
,ifnull(DAILY_PEOPLE_VACCINATED_PER_HUNDRED,previousnonnull_dpvph) DAILY_PEOPLE_VACCINATED_PER_HUNDRED
from
(
SELECT
     DATE_REPORTED
    ,COUNTRY_CODE
    ,COUNTRY
    ,NEW_CASES
    ,CUMULATIVE_CASES                          --ifnull(,previousnonnull_tv)
    ,NEW_DEATHS
    ,CUMULATIVE_DEATHS
    ,TOTAL_VACCINATIONS
    ,PEOPLE_VACCINATED
    ,PEOPLE_FULLY_VACCINATED
    ,TOTAL_BOOSTERS
    ,DAILY_VACCINATIONS
    ,TOTAL_VACCINATIONS_PER_HUNDRED
    ,PEOPLE_VACCINATED_PER_HUNDRED
    ,PEOPLE_FULLY_VACCINATED_PER_HUNDRED
    ,TOTAL_BOOSTERS_PER_HUNDRED
    ,DAILY_VACCINATIONS_PER_MILLION
    ,DAILY_PEOPLE_VACCINATED
    ,DAILY_PEOPLE_VACCINATED_PER_HUNDRED
    ,LAG(daily_vaccinations) IGNORE NULLS OVER (ORDER BY date_reported) previousnonnull_dv
    ,LAG(total_vaccinations) IGNORE NULLS OVER (ORDER BY date_reported) previousnonnull_tv
    ,LAG(people_vaccinated) IGNORE NULLS OVER (ORDER BY date_reported) previousnonnull_pv
    ,LAG(people_fully_vaccinated) IGNORE NULLS OVER (ORDER BY date_reported) previousnonnull_pfv
    ,LAG(total_boosters) IGNORE NULLS OVER (ORDER BY date_reported) previousnonnull_tb
    ,LAG(TOTAL_VACCINATIONS_PER_HUNDRED) IGNORE NULLS OVER (ORDER BY date_reported) previousnonnull_tvph
    ,LAG(PEOPLE_VACCINATED_PER_HUNDRED) IGNORE NULLS OVER (ORDER BY date_reported) previousnonnull_pvph
    ,LAG(PEOPLE_FULLY_VACCINATED_PER_HUNDRED) IGNORE NULLS OVER (ORDER BY date_reported) previousnonnull_pfvph
    ,LAG(TOTAL_BOOSTERS_PER_HUNDRED) IGNORE NULLS OVER (ORDER BY date_reported) previousnonnull_tbph
    ,LAG(DAILY_VACCINATIONS_PER_MILLION) IGNORE NULLS OVER (ORDER BY date_reported) previousnonnull_dvpm
    ,LAG(DAILY_PEOPLE_VACCINATED) IGNORE NULLS OVER (ORDER BY date_reported) previousnonnull_dpv
     ,LAG(DAILY_PEOPLE_VACCINATED_PER_HUNDRED) IGNORE NULLS OVER (ORDER BY date_reported) previousnonnull_dpvph
    

FROM covid_data_processing));

--select * from data_process

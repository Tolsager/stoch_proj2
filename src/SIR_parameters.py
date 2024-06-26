# Rates for specific diseases (per day)  

# Natural causes rates
# In 2023 57500 were born in denmark (with population 5.9 million) 
# And the expected life span is 81 years 
natural_causes_rates = {"Birth": 57_500 / (5_900_000 * 365), "Death": 1/(81 * 365)} 

# Vaccination rates
vaccination_rates = {"Vaccination": 100, "Vaccination1 I": 5/10, "Vaccination2 I": 1/10}

# Corona data
# Corona rates 1 - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8642156/
corona_rates = {"Infection": 0.17,  # Infection without any precautions (from the link above)
                "Infection2": 0.13, # Infection with small precautions
                "Infection3": 0.10, # Infection with drastic precautions
                "Recovery": 0.082,  # From the link above
                "Death General": 0.02,      # General statistic when not distinguishing between severe and non-severe cases
                "Death": 0.001,  # Death when having mild corona 
                "Death Severe": 0.5, # Death when having severe corona without hospitalization
                "Death Treatment": 0.1, # Death when having severe corona with hospitalization
                "Reinfection": 1/60, # Reinfection rates could not be found, but we say that on average you can get reinfected every 2 months 
                "Severe": 130/3300, # Severe cases is calculated from 130_000 hospitalizations, and 3_300_000 cases. 
                "Severe Recovery": 3, # Times longer recovery when having servere corona
                "Severe Recovery Treatment": 2, # Times longer recovery when having servere corona and hospitalized
                }

# Corona rates 2 - https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265815
# corona_rates = {"Infection": 0.5,  # Infection without any precautions (from the link above)
#                 "Infection2": 0.25, # Infection with small precautions
#                 "Infection3": 0.10, # Infection with drastic precautions
#                 "Recovery": 0.13,  # From the link above
#                 "Death General": 0.02,      # General statistic when not distinguishing between severe and non-severe cases
#                 "Death": 0.002,  # Death when having mild corona 
#                 "Death Severe": 0.5, # Death when having severe corona without hospitalization
#                 "Death Treatment": 0.1, # Death when having severe corona with hospitalization
#                 "Reinfection": 1/60, 
#                 "Severe": 130/3300, 
#                 "Severe Recovery": 3, # Times longer recovery when having servere corona
#                 "Severe Recovery Treatment": 2, # Times longer recovery when having servere corona and hospitalized
#                 }

N = 50_000
t_max = 365*3
I0 = 100
HOSPITAL_CAPACITY = 0.0025*N

# For birth and death
covid_birth_death =  [corona_rates["Infection"], # Infection rate
                    corona_rates["Recovery"], # Recovery rate
                    corona_rates["Death General"], # Death rate frome disease 
                    natural_causes_rates["Birth"], # Birth rate 
                    natural_causes_rates["Death"], # Death rate from natural causes
]

# For birth and death with reinfection
extended_covid = [corona_rates["Infection"], # Infection rate
                    corona_rates["Recovery"], # Recovery rate
                    corona_rates["Death General"], # Death rate frome disease 
                    corona_rates["Reinfection"], # Reinfection rate
                    natural_causes_rates["Birth"], # Birth rate 
                    natural_causes_rates["Death"], # Death rate from natural causes
]

# For birth and death with reinfection and severe cases of disease
covid_severe_disease = [corona_rates["Infection"], # Infection rate
                    corona_rates["Recovery"], # Recovery rate
                    corona_rates["Death"], # Death rate frome disease 
                    corona_rates["Severe"], # Severe rate
                    corona_rates["Recovery"]/corona_rates["Severe Recovery"], # Severe recovery rate
                    corona_rates["Death Severe"], # Severe death rate
                    corona_rates["Reinfection"], # Reinfection rate
                    natural_causes_rates["Birth"], # Birth rate 
                    natural_causes_rates["Death"], # Death rate from natural causes
]

# For birth and death with reinfection, severe cases of disease and vaccination
covid_vax = [corona_rates["Infection"], # Infection rate
            corona_rates["Recovery"], # Recovery rate
            corona_rates["Death"], # Death rate frome disease 
            corona_rates["Death Severe"], # Death rate frome disease when being severly ill
            corona_rates["Death Treatment"], # Death rate frome disease when being severly ill and hospitalized
            corona_rates["Severe"], # Severe rate
            corona_rates["Recovery"]/corona_rates["Severe Recovery"], # Severe recovery rate
            corona_rates["Recovery"]/corona_rates["Severe Recovery Treatment"], # Severe recovery rate when hospitalized
            corona_rates["Reinfection"], # Reinfection rate
            vaccination_rates["Vaccination"], # Vaccination rate
            0, # This is a filler because we have a bonus event 
            vaccination_rates["Vaccination1 I"], # Infection rate for vaccinated round 1
            vaccination_rates["Vaccination2 I"], # Infection rate for vaccinated round 2
            natural_causes_rates["Birth"], # Birth rate 
            natural_causes_rates["Death"], # Death rate from natural causes
]


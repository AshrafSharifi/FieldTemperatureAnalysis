import pandas as pd


if __name__ == "__main__":
    #config = 4
    #plot_traj(config)       
    # df = pd.read_csv('sensor_t1.csv')
    # df.head()
    # df.rename(columns = {'temp_to_estimate' :'temp_to_estimate', 'distanza_da_centralina_cm':'dist_to_central_station', 'anno_acquisizione':'year', 
    #                            'mese_acquisizione':'month', 
    #                            'settimana_acquisizione_su_anno':'week',
    #                            'giorno_acquisizione_su_anno':'day_of_year',
    #                            'giorno_acquisizione_su_mese':'day_of_month',
    #                            'giorno_acquisizione_su_settimana':'day_of_week',
    #                            'ora_acquisizione':'hour',
    #                            'timestamp_normalizzato':'complete_timestamp(YYYY_M_DD_HH_M)'}, inplace = True)

                               
    # df.to_csv("sensor_t1.csv")
    
    df = pd.read_csv('data/sensor_t8.csv',names=[
     'temp_to_estimate',
     'dist_to_central_station',
     'year',
     'month',
     'week',
     'day_of_year',
     'day_of_month',
     'day_of_week',
     'hour',
     'complete_timestamp(YYYY_M_DD_HH_M)',
     'barometer_hpa',
     'temp_centr',
     'hightemp_centr',
     'lowtemp_centr',
     'hum',
     'dewpoint__c',
     'wetbulb_c',
     'windspeed_km_h',
     'windrun_km',
     'highwindspeed_km_h',
     'windchill_c',
     'heatindex_c',
     'thwindex_c',
     'thswindex_c',
     'rain_mm',
     'rain_rate_mm_h',
     'solar_rad_w_m_2',
     'solar_energy_ly',
     'high_solar_rad_w_m_2',
     'ET_Mm',
     'heating_degree_days',
     'cooling_degree_days',
     'humidity_rh',
     'solar_klux']);
    df.to_csv("data/sensor_t8.csv")
    # for i in range(2:8):
    #     df = pd.read_csv('sensor_t'+i+'.csv');
    #     for j in range(1:34):
            
        
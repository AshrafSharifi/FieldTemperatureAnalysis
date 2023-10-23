import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from functions import *

def first_derivative(data, threshold, date):
    temperature_data = data[:, 0]
    time_vector = data[:, 1]
    
    # Calculate the first derivative (rate of change)
    derivative = np.diff(temperature_data)
    
    # Find indices where the derivative exceeds the threshold
    sharp_change_indices = np.where(np.abs(derivative) > threshold)[0]
    
    # Display the sharp change indices and values
    print('Sharp Change Indices:')
    print(time_vector[sharp_change_indices])
    print('Sharp Change Values:')
    print(derivative[sharp_change_indices])
    
    # Plot the temperature data with detected sharp changes
    plt.figure()
    plt.plot(time_vector, temperature_data, 'b', linewidth=2)
    plt.scatter(time_vector[sharp_change_indices], temperature_data[sharp_change_indices], c='r', marker='o')
    plt.xlabel('Time Intervals')
    plt.ylabel('Temperature')
    plt.title("Temperature Data with Sharp Change Detection (" + date + ")")
    plt.legend(['Temperature Data', 'Sharp Changes'])
    plt.grid(True)
    plt.show()
    
def find_sharp_changes(data):

   temperature_data = data["temp_to_estimate"]
   time_vector = data["hour"]
   
   # Calculate the whole average temperature
   average_temperature = temperature_data.mean()
   
   # Calculate the standard deviation of the temperature
   temperature_standard_deviation = temperature_data.std()
   
   # Set a threshold for detecting sharp changes in temperature
   threshold = 2 * temperature_standard_deviation
   
   # Identify any temperature changes that fall outside of the typical range
   df_outliers = (data[temperature_data > average_temperature + threshold])|(data[temperature_data < average_temperature - threshold])

   
   
   # if self.displayResult:
   #     # Display the sharp change indices and values
   #     print('Sharp Change Indices:')
   #     print(time_vector[sharp_change_indices])
   #     print('Sharp Change Values:')
   #     print(derivative[sharp_change_indices])
       
   #     # Plot the temperature data with detected sharp changes
   #     plt.figure()
   #     plt.plot(time_vector, temperature_data, 'b', linewidth=2)
   #     plt.scatter(time_vector[sharp_change_indices], temperature_data[sharp_change_indices], c='r', marker='o')
   #     plt.xlabel('Time Intervals')
   #     plt.ylabel('Temperature')
   #     plt.title("Temperature Data with Sharp Change Detection (" + self.date + ")")
   #     plt.legend(['Temperature Data', 'Sharp Changes'])
   #     plt.grid(True)
   #     plt.show()
   # if self.hasOutput:
   #     out= {}
   #     out["SharpChangeValues"] = derivative[sharp_change_indices]
   #     out["SharpChangeIndices"] = sharp_change_indices
   #     out["SharpChangeTimes"] = time_vector[sharp_change_indices]
   #     return out    
def corr_matrix(data):
    df = data[['temperatureChange','temp_to_estimate','barometer_hpa','temp_centr','hightemp_centr','lowtemp_centr','hum','dewpoint__c','wetbulb_c','windspeed_km_h','windchill_c','heatindex_c','thswindex_c','rain_mm','solar_rad_w_m_2','solar_energy_ly','humidity_rh']]
    # Drop non-numerical variables
    _, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    corr_matrix = df.corr()
    display(corr_matrix)
    
    sns.heatmap(corr_matrix,ax=axes);

if __name__ == '__main__':

    sensor_number = 1
    pd.options.mode.chained_assignment = None 
    # Time period
    year = 2022
    month = 1
    day_of_month = 9
    
    # Define a threshold to detect sharp changes
    threshold = 2.69
    
    sensor_data = pd.read_csv("../data/sensor_t" + str(sensor_number) + ".csv")
    

    # Calculate temperature change as the difference between consecutive temperature values
    sensor_data['temperatureChange'] = sensor_data['temp_to_estimate'].diff()
    
    # The first row will have NaN in the 'TemperatureChange' column; you can replace it with 0 if needed
    sensor_data['temperatureChange'].fillna(0, inplace=True)
    
    # Filter data based on the specified date
    idx = (sensor_data['year'] == year) & (sensor_data['month'] == month) & (sensor_data['day_of_month'] == day_of_month)
    filtered_data = sensor_data.loc[idx, ['temp_to_estimate', 'hour','temperatureChange']]
    filtered_data = filtered_data.sort_values(by='hour')
    
    find_sharp_changes(filtered_data)
    filtered_data = filtered_data.to_numpy()
    

    
    date = str(year) + '-' + str(month) + '-' + str(day_of_month)
    first_derivative(filtered_data, threshold, date)
    corr_matrix(sensor_data.loc[idx,:])
    
    
    
    

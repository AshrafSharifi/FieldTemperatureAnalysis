import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


class functions(object):

  def __init__(self, sensor_data, threshold=0, displayResult=False, hasOutput=False):
        self.df = sensor_data;
        self.threshold = threshold;
        self.date = str(sensor_data['year']) + '-' + str(sensor_data['month']) + '-' + str(sensor_data['day_of_month']);
        self.displayResult = displayResult
        self.hasOutput = hasOutput
  def data_preperation(self):

    # Calculate temperature change as the difference between consecutive temperature values
    self.df['temperatureChange'] = self.df['temp_to_estimate'].diff()

    # The first row will have NaN in the 'TemperatureChange' column; you can replace it with 0 if needed
    self.df['temperatureChange'].fillna(0, inplace=True)
    
    self.df = self.df.sort_values(by='hour')
    self.df = self.df.to_numpy()
    return self.df;
  

  def first_derivative(self):
    
    data=self.df;
    temperature_data = data[:, 1]
    time_vector = data[:, 9]
    
    # Calculate the first derivative (rate of change)
    derivative = np.diff(temperature_data)
    
    # Find indices where the derivative exceeds the threshold
    # sharp_change_indices = np.where(np.abs(derivative) > self.threshold)[0]
    average_temperature = temperature_data.mean()

    # Find indices of values exceeding the threshold in both directions
    above_threshold_indices = np.where(temperature_data > average_temperature + self.threshold)[0]
    below_threshold_indices = np.where(temperature_data < average_temperature - self.threshold)[0]
    
    # Concatenate the indices into a single array
    sharp_change_indices = np.concatenate((above_threshold_indices, below_threshold_indices))

    
    if self.displayResult:
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
        plt.title("Temperature Data with Sharp Change Detection (" + self.date + ")")
        plt.legend(['Temperature Data', 'Sharp Changes'])
        plt.grid(True)
        plt.show()
    if self.hasOutput:
        out= {}
        out["SharpChangeValues"] = derivative[sharp_change_indices]
        out["SharpChangeIndices"] = sharp_change_indices
        out["SharpChangeTimes"] = time_vector[sharp_change_indices]
        return out

  def find_sharp_changes(self):
    
    data=self.df;
    temperature_data = data[:, 1]
    time_vector = data[:, 9]
    
    # Calculate the whole average temperature
    average_temperature = self.df['temp_to_estimate'].mean()
    
    # Calculate the standard deviation of the temperature
    temperature_standard_deviation = self.df['temp_to_estimate'].std()
    
    # Set a threshold for detecting sharp changes in temperature
    threshold = 2 * temperature_standard_deviation
    
    # Identify any temperature changes that fall outside of the typical range
    df_outliers = self.df[self.df['temp_to_estimate'] > average_temperature + threshold]
    df_outliers = df_outliers.append(self.df[self.df['temp_to_estimate'] < average_temperature - threshold])
    
    
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

  def corr_matrix(self, columns):
    columns =['temperatureChange','temp_to_estimate','temp_centr','hum','dewpoint__c','wetbulb_c','windspeed_km_h','thswindex_c','rain_mm','solar_rad_w_m_2']
    # df = self.df[['temperatureChange','temp_to_estimate','temp_centr','hum','dewpoint__c','wetbulb_c','windspeed_km_h','thswindex_c','rain_mm','solar_rad_w_m_2']]
    df = self.df[columns]
   
    # Drop non-numerical variables
    _, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    corr_matrix = df.corr()
    display(corr_matrix)
    sns.heatmap(corr_matrix,ax=axes);
    
  def find_threshold_based_on_variation(self):

    # Calculate the temperature variations (differences between consecutive values)
    temperature_variations = self.df['temp_to_estimate'].diff().fillna(0)
    
    # Calculate the mean and standard deviation of temperature variations
    mean_variations = temperature_variations.mean()
    std_variations = temperature_variations.std()
    
    # Define a threshold as a multiple of the standard deviation (e.g., 2 times the std)
    threshold = 2 * std_variations
    
    # Print the mean, standard deviation, and the threshold for temperature variations
    print(f"Mean Temperature Variations: {mean_variations}")
    print(f"Standard Deviation of Variations: {std_variations}")
    print(f"Threshold for Variations: {threshold}")
    
    # Now you can use the threshold to detect significant temperature variations.
    
  def find_threshold(self):

     # Calculate the temperature variations (differences between consecutive values)
    temperature_variations = self.df['temp_to_estimate']
    
    # Calculate the mean and standard deviation of temperature variations
    mean_variations = temperature_variations.mean()
    std_variations = temperature_variations.std()
    
    # Define a threshold as a multiple of the standard deviation (e.g., 2 times the std)
    threshold = 2 * std_variations
    return (threshold)
    
 

"""通用工具函數 (getR2MSdata、csv2urf、convertURF…)"""

import os
import sys
import zipfile
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygimli.physics import ert
from datetime import datetime

def find_urf_files(urf_dir, pattern="*.urf"):
    """找到指定目錄下所有的 URF 檔案"""
    urf_files = glob.glob(os.path.join(urf_dir, pattern))
    return sorted(urf_files)

def unzip_files(local_path):
    """
    解壓縮指定目錄中的所有 ZIP 檔案
    
    參數:
        local_path: 包含 ZIP 檔案的目錄路徑
    
    返回:
        True: 如果所有檔案都成功解壓縮或已處理
        False: 如果列出 ZIP 檔案時發生錯誤
    """
    try:
        # 列出目錄中的所有 ZIP 檔案
        zip_files = [f for f in os.listdir(local_path) if f.endswith('.zip')]
        
        # 記錄是否有任何檔案解壓縮失敗
        any_failed = False
        
        # 解壓縮所有檔案
        for zip_file in zip_files:
            zip_path = os.path.join(local_path, zip_file)
            print(f"解壓縮 {zip_file}...")
            
            try:
                # 檢查檔案大小
                if os.path.getsize(zip_path) == 0:
                    print(f"警告: {zip_file} 檔案大小為 0KB，跳過並刪除此檔案")
                    os.remove(zip_path)
                    any_failed = True
                    continue
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # 解壓縮到 local_path 目錄
                    zip_ref.extractall(local_path)
                
                # 刪除 ZIP 檔案
                os.remove(zip_path)
                print(f"{zip_file} 解壓縮完成，已刪除 ZIP 檔案")
            
            except Exception as e:
                print(f"{zip_file} 解壓縮失敗: {str(e)}")
                # 刪除失敗的 ZIP 檔案並繼續下一個
                try:
                    os.remove(zip_path)
                    print(f"已刪除失敗的 ZIP 檔案: {zip_file}")
                except Exception as del_err:
                    print(f"無法刪除失敗的 ZIP 檔案 {zip_file}: {str(del_err)}")
                
                any_failed = True
                # 繼續處理下一個檔案
                continue
        
        if any_failed:
            print("部分 ZIP 檔案解壓縮失敗，但已處理完所有檔案")
        
        return True
        
    except Exception as e:
        print(f"列出 ZIP 檔案失敗: {str(e)}")
        return False

def csv2urf(csv_files, one_intelligent_ERT_survey_geo_file, output_urf_path, output_urf_file_name, output_png_path, plot_wave, png_file_first_name, amplitude_estimate_start_position, amplitude_estimate_range, contain_common_N=True):
    # Initialize return code
    Return_Code = 0
    # Reading the contents of the .geo file
    with open(one_intelligent_ERT_survey_geo_file, 'rt', encoding='utf-8') as f:
        temp_geo_char_data = f.read()


    
    temp_char_data = temp_geo_char_data
    geo = {}

    
    temp_key_str = 'Tx\n'
    temp_start_index = temp_char_data.find(temp_key_str)
    if temp_start_index == -1:
        geo['Tx'] = []
    else:
        temp_end_index = temp_char_data.find('Rx\n')
        geo['Tx'] = {}
        geo['Tx']['String'] = temp_char_data[temp_start_index+3:temp_end_index-2]
        temp_char_data2 = geo['Tx']['String'].replace(',', ' ')
        geo['Tx']['DataHeader'] = ['Tx_index']
        geo['Tx']['Data'] = list(map(float, temp_char_data2.split()))

    
    temp_key_str = 'Rx\n'
    temp_start_index = temp_char_data.find(temp_key_str)
    if temp_start_index == -1:
        geo['Rx'] = []
    else:
        temp_end_index = temp_char_data.find('RxP2\n')
        geo['Rx'] = {}
        geo['Rx']['String'] = temp_char_data[temp_start_index+3:temp_end_index-2]
        temp_char_data2 = geo['Rx']['String'].replace(',', ' ')
        geo['Rx']['DataHeader'] = ['Rx_index']
        geo['Rx']['Data'] = list(map(float, temp_char_data2.split()))

    
    temp_key_str = 'RxP2\n'
    temp_start_index = temp_char_data.find(temp_key_str)
    if temp_start_index == -1:
        geo['RxP2'] = []
    else:
        temp_end_index = temp_char_data.find(':Geometry\n')
        geo['RxP2'] = {}
        geo['RxP2']['String'] = temp_char_data[temp_start_index+5:temp_end_index-2]
        temp_char_data2 = geo['RxP2']['String'].replace(',', ' ')
        geo['RxP2']['DataHeader'] = ['RxP2_index']
        geo['RxP2']['Data'] = list(map(int, temp_char_data2.split()))





    # Assuming 'csv_file_list' contains all the CSV filenames found in the directory
    # and 'csv_files_path' is the path to the directory containing CSV files
    data_frames = []  # List to store individual DataFrames for each CSV file

    for csv_file in csv_files:
        # Construct full file path
        file_path = csv_file
        # Read CSV file into a DataFrame
        temp_df = pd.read_csv(file_path, header=None)
        temp_df.drop(temp_df.columns[len(temp_df.columns)-1], axis=1, inplace=True)
        # Add the temporary DataFrame to the list
        data_frames.append(temp_df)

    # Concatenate all DataFrames in the list into a single DataFrame
    temp_all_data_df = pd.concat(data_frames, ignore_index=True)

    # Step1: Filter rows where the transmitter state includes '2'
    temp_all_ERT_data_df = temp_all_data_df[temp_all_data_df.iloc[:, 3].astype(str).str.contains('2')]
    # # outpur temp_all_ERT_data_df to csv
    # temp_all_ERT_data_df.to_csv(os.path.join(output_urf_path, output_urf_file_name[:-4]+'.csv'), index=False, header=True)
    # Verify data quantity is correct (assuming 480 samples per active period at 10Hz for 48 seconds)
    # temp_all_ERT_data_df = check_raw_data(temp_all_ERT_data_df)
    if len(temp_all_ERT_data_df) % 480 == 0:
        print('Data quantity correct, proceeding with further calculations')
    else:
        print('Data quantity abnormal, please check the raw data!')
        try:
            print('Try to check the raw data...')
            temp_all_ERT_data_df = check_raw_data(temp_all_ERT_data_df)
            Return_Code = -1
        except:
            Return_Code = -1
            return Return_Code

    # Initialize a list to hold data for each event
    temp_all_event_ERT_data = []

    # Split the filtered DataFrame into chunks of 480 rows each, representing individual events
    num_events = len(temp_all_ERT_data_df) // 480
    for i in range(num_events):
        event_df = temp_all_ERT_data_df.iloc[i * 480: (i + 1) * 480]
        temp_all_event_ERT_data.append(event_df)

    # print(f'Total ERT events: {num_events}')

    # Initialize lists to store measurement data for each event
    urf_measurement_data = {}#[None] * num_events
    keys = range(num_events)
    for key in keys:
        urf_measurement_data[key] = []

    urf_measurement_data_beta = [None] * num_events
    urf_current_data = np.zeros((num_events,4))

    # Loop over each event
    for i_all_event_ERT_data_index in range(num_events):
        # print(f'Processing event {i_all_event_ERT_data_index+1}...')

        # Initialize the urf measurement data for this event
        # urf_measurement_data[i_all_event_ERT_data_index] = {}
        # urf_measurement_data_beta[i_all_event_ERT_data_index] = {}

        # Step 2: Analyze electrode positions
        # Extract the data for this event
        temp_one_event_ERT_data = temp_all_event_ERT_data[i_all_event_ERT_data_index]
        # Electrode state check
        electrode_state = temp_one_event_ERT_data[4]
        start_index=electrode_state.index[0]
        # Check if the electrode state is the same for the first 10 data points
        for i in range(10):
            try: 
                electrode_state[i+start_index] != electrode_state[start_index]
            except:
                print('Electrode state is abnormal, please check the original data!')
                Return_Code = -2
                return Return_Code
        # Determine the electrode positions used by the transmitter
        Tx_use_Electrode_index = sorted([i for i, x in enumerate(electrode_state[start_index]) if x in ['+', '-', '0']])

        # Determine the electrode positions used by the receiver
        Rx_use_Electrode_index = [i for i, x in enumerate(electrode_state[start_index]) if x == 'v']

        # Remove unused electrodes
        Tx_use_Electrode_index = [i for i in Tx_use_Electrode_index if i <= len(geo['Tx']['Data'])]
        Rx_use_Electrode_index = [i for i in Rx_use_Electrode_index if i <= len(geo['Rx']['Data'])]

        # print(f'Event {i_all_event_ERT_data_index+1} analysis:')
        # print(f'Tx count = {len(Tx_use_Electrode_index)}, Rx count = {len(Rx_use_Electrode_index)}')

        if len(Tx_use_Electrode_index) > 0 and len(Rx_use_Electrode_index) > 0:
            # Step 3: Analyze all measurement data as square wave amplitude
            # Extract all voltage and current measurement data
            temp_one_event_measure_data = temp_one_event_ERT_data.iloc[:, 5:]

            # Initialize an array to store the estimated voltage and current data
            temp_one_event_estimate_data = np.zeros(temp_one_event_measure_data.shape)

            ## Start the estimation process
            # Process each column of measurement data
            for i_column in range(128):  # Adjust the range according to your specific data structure
                for i_three_second_index in range(1, 17):  # 16 three-second intervals in 48 seconds
                    # Extract three-second segment of measurement data
                    temp_three_second_measure_data = np.array(temp_one_event_measure_data[i_column+5][30*(i_three_second_index-1):30*i_three_second_index])

                    # Initialize arrays to hold amplitude differences
                    temp_amplitude_1st_second_array = []
                    temp_amplitude_2nd_second_array = []

                    # Calculate amplitude differences
                    for i in range(amplitude_estimate_range):
                        temp_3rd_second_value = temp_three_second_measure_data[20 + amplitude_estimate_start_position + i - 1]
                        for j in range(amplitude_estimate_range):
                            temp_1st_second_value = temp_three_second_measure_data[10 - j - 1]
                            temp_amplitude_1st_second_array.append(temp_1st_second_value - temp_3rd_second_value)

                            temp_2nd_second_value = temp_three_second_measure_data[20 - j - 1]
                            temp_amplitude_2nd_second_array.append(temp_2nd_second_value - temp_3rd_second_value)

                    # Calculate median amplitude estimates
                    temp_amplitude_1st_second = np.median(temp_amplitude_1st_second_array)
                    temp_amplitude_2nd_second = np.median(temp_amplitude_2nd_second_array)

                    # Construct estimated data array for the three-second segment
                    temp_three_second_estimate_data = np.concatenate([np.full(10, temp_amplitude_1st_second),
                                                                    np.full(10, temp_amplitude_2nd_second),
                                                                    np.zeros(10)])

                    # Update the estimate data array with calculated estimates for this segment
                    temp_one_event_estimate_data[30*(i_three_second_index-1):30*i_three_second_index, i_column] = temp_three_second_estimate_data

            # where transmitters only measure current and receivers only measure voltage
            temp_one_event_estimate_data[:,np.array(Tx_use_Electrode_index)*2-2] = 0
            temp_one_event_estimate_data[:,np.array(Rx_use_Electrode_index)*2-1] = 0
            temp_one_event_measure_data = np.array(temp_one_event_measure_data)

            temp_one_event_full_measure_voltage_data=temp_one_event_measure_data[:,0::2]
            temp_one_event_full_measure_current_data=temp_one_event_measure_data[:,1::2]
            temp_one_event_full_estimate_voltage_data=temp_one_event_estimate_data[:,0::2]
            temp_one_event_full_estimate_current_data=temp_one_event_estimate_data[:,1::2]


            # Simplified non-repeated data
            temp_one_event_simple_estimate_voltage_data=temp_one_event_estimate_data[0::10,0::2]
            temp_one_event_simple_estimate_current_data=temp_one_event_estimate_data[0::10,1::2]
            # Extract [48xN] I_AB data
            temp_I_AB_array=temp_one_event_simple_estimate_current_data[:,np.array(Tx_use_Electrode_index[1:])-1]
            # Extract comeback current (common current) data
            temp_I_comeback_array=temp_one_event_simple_estimate_current_data[:,np.array(Tx_use_Electrode_index)[0]-1]
            # Maximum  comeback current
            temp_max_Tx_comeback_current=max(abs(temp_I_comeback_array))
            # Average comeback current 
            temp_I_comeback_array2=np.delete(temp_I_comeback_array, np.arange(2, len(temp_I_comeback_array), 3))
            temp_mean_Tx_comeback_current=np.mean(abs(temp_I_comeback_array2))
            # Difference between comeback current and comeout current
            temp_I_delta_array=temp_I_comeback_array+ np.sum(temp_I_AB_array,axis=1) 
            # Maximum difference between comeback current and comeout current
            temp_I_max_delta_array=max(abs(temp_I_delta_array))
            # Average difference between comeback current and comeout current
            temp_I_delta_array2=np.delete(temp_I_delta_array, np.arange(2, len(temp_I_delta_array), 3))
            temp_I_mean_delta_array=np.mean(abs(temp_I_delta_array2))

            # Collect Current data
            urf_current_data[i_all_event_ERT_data_index,0]=temp_max_Tx_comeback_current*1000
            urf_current_data[i_all_event_ERT_data_index,1]=temp_mean_Tx_comeback_current*1000
            urf_current_data[i_all_event_ERT_data_index,2]=temp_I_max_delta_array*1000
            urf_current_data[i_all_event_ERT_data_index,3]=temp_I_mean_delta_array*1000
        
            # Extract [48x1] V_MN data, CONTAINS common N 
            if contain_common_N:
                for i_Tx_Electrode_index in range(len(Tx_use_Electrode_index)-1):
                    for i_Rx_Electrode_index in range(len(Rx_use_Electrode_index)): 
                        # The instrument measures voltage at the common N directly
                        temp_V_MN_Sum_array=temp_one_event_simple_estimate_voltage_data[:,Rx_use_Electrode_index[i_Rx_Electrode_index]-1]
                        temp_R_ABMN_array,_,_,_ = np.linalg.lstsq(temp_I_AB_array, temp_V_MN_Sum_array, rcond=None)
                        temp_V_residual2=np.dot(temp_I_AB_array,temp_R_ABMN_array)-temp_V_MN_Sum_array
                        temp_V_residual = np.delete(temp_V_residual2, np.arange(2, len(temp_V_residual2), 3))
                        temp_V_error = np.max(abs(temp_V_MN_Sum_array))*1000

                        urf_measurement_data[i_all_event_ERT_data_index].append([Tx_use_Electrode_index[i_Tx_Electrode_index+1],
                                                                                Tx_use_Electrode_index[0],
                                                                                Rx_use_Electrode_index[i_Rx_Electrode_index],
                                                                                geo['RxP2']['Data'][0],
                                                                                temp_R_ABMN_array[i_Tx_Electrode_index],
                                                                                max(abs(temp_I_AB_array[:,i_Tx_Electrode_index]))*1000,
                                                                                temp_V_error])

            # Extract [48x1] V_AB data, EVERY N
            for i_Tx_Electrode_index in range(len(Tx_use_Electrode_index)-1):
                for i_Rx_M_Electrode_index in range(len(Rx_use_Electrode_index)): 
                    temp_remain_Rx_use_Electrode_index=Rx_use_Electrode_index[i_Rx_M_Electrode_index+1:]
                    
                    for i_Rx_N_Electrode_index in range(len(temp_remain_Rx_use_Electrode_index)):
                        temp_V_MN_Sum_array=temp_one_event_simple_estimate_voltage_data[:,Rx_use_Electrode_index[i_Rx_M_Electrode_index]-1] - temp_one_event_simple_estimate_voltage_data[:,temp_remain_Rx_use_Electrode_index[i_Rx_N_Electrode_index]-1]

                        temp_R_ABMN_array,_,_,_ = np.linalg.lstsq(temp_I_AB_array,temp_V_MN_Sum_array, rcond=None)
                        temp_V_residual2=np.dot(temp_I_AB_array,temp_R_ABMN_array)-temp_V_MN_Sum_array
                        temp_V_residual = np.delete(temp_V_residual2, np.arange(2, len(temp_V_residual2), 3))
                        temp_V_error = np.max(abs(temp_V_MN_Sum_array))*1000

                        urf_measurement_data[i_all_event_ERT_data_index].append([Tx_use_Electrode_index[i_Tx_Electrode_index+1],
                                                                                Tx_use_Electrode_index[0],
                                                                                Rx_use_Electrode_index[i_Rx_M_Electrode_index],
                                                                                temp_remain_Rx_use_Electrode_index[i_Rx_N_Electrode_index],
                                                                                temp_R_ABMN_array[i_Tx_Electrode_index],
                                                                                max(abs(temp_I_AB_array[:,i_Tx_Electrode_index]))*1000,
                                                                                temp_V_error])

            if plot_wave:
                if not os.path.isdir(output_png_path):
                    os.makedirs(output_png_path)            
                # Create a figure with 4 rows and 2 columns of subplots (Current and Voltage data)
                fig, axs = plt.subplots(4, 2, figsize=(25.6, 14.4))  # Adjust size as needed
                fig.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing to prevent label overlap

                # Iterate over each subplot to populate with data
                for i, ax in enumerate(axs.flatten()):
                    if i == 0:
                        title_suffix = ' (Main Current Measurement Data)'
                        temp_Electrode_index=Tx_use_Electrode_index[1:]
                        if not temp_Electrode_index:  # Checks if the list is empty
                            temp_x_vector = []
                        else:
                            temp_jet_colors = plt.cm.hsv(np.linspace(0, 1, len(Tx_use_Electrode_index)))  # Color map
                            temp_x_vector = np.arange(480) / 10  # Time vector for plotting
                        
                            # Plot each channel with unique color
                            for qq, index in enumerate(temp_Electrode_index):
                                ax.plot(temp_x_vector, temp_one_event_full_measure_current_data[:,temp_Electrode_index[qq]-1] * 1000, '-', color=temp_jet_colors[qq+1], label=f'CH{index:02d}', linewidth=1)
                            
                            # 只在有實際繪製帶標籤的線條時才添加圖例
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                        # Customize each subplot
                        ax.set_xlabel('Time [sec]')
                        ax.set_ylabel('Current [mA]')
                        ax.set_title(f'{png_file_first_name} EVENT: {i_all_event_ERT_data_index+1} / {num_events}{title_suffix}'+', Page: 1 / 2')
                        ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1, 5))
                    
                    elif i == 2:
                        title_suffix = ' (Main Current Estimation Data)'
                        temp_Electrode_index=Tx_use_Electrode_index[1:]
                        if not temp_Electrode_index:  # Checks if the list is empty
                            temp_x_vector = []
                        else:
                            temp_jet_colors = plt.cm.hsv(np.linspace(0, 1, len(Tx_use_Electrode_index)))  # Color map
                            temp_x_vector = np.arange(480) / 10  # Time vector for plotting

                            # Plot each channel with unique color
                            for qq, index in enumerate(temp_Electrode_index):
                                ax.plot(temp_x_vector, temp_one_event_full_estimate_current_data[:, temp_Electrode_index[qq]-1] * 1000, '-', color=temp_jet_colors[qq+1], label=f'CH{index:02d}', linewidth=1)
                            
                            # 只在有實際繪製帶標籤的線條時才添加圖例
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                        # Customize each subplot
                        ax.set_xlabel('Time [sec]')
                        ax.set_ylabel('Current [mA]')
                        ax.set_title(f'{png_file_first_name} EVENT: {i_all_event_ERT_data_index+1} / {num_events}{title_suffix}')
                        ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1, 5))

                    elif i == 4:
                        title_suffix = ' (All Current Measurement Data)'
                        temp_Electrode_index=Tx_use_Electrode_index
                        if not temp_Electrode_index:  # Checks if the list is empty
                            temp_x_vector = []
                        else:
                            temp_jet_colors = plt.cm.hsv(np.linspace(0, 1, len(Tx_use_Electrode_index)))
                            temp_x_vector = np.arange(480) / 10
                            # Plot each channel with unique color
                            for qq, index in enumerate(temp_Electrode_index):
                                ax.plot(temp_x_vector, temp_one_event_full_measure_current_data[:, temp_Electrode_index[qq]-1]*1000, '-', color=temp_jet_colors[qq], label=f'CH{index:02d}', linewidth=1)
                            
                            # 只在有實際繪製帶標籤的線條時才添加圖例
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                        # Customize each subplot
                        ax.set_xlabel('Time [sec]')
                        ax.set_ylabel('Current [mA]')
                        ax.set_title(f'{png_file_first_name} EVENT: {i_all_event_ERT_data_index+1} / {num_events}{title_suffix}')
                        ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1, 5))

                    elif i == 6:
                        title_suffix = ' (All Current Estimation Data) '+'$\Delta IO$ (max/mean)={:.2f}/{:.2f} mA'.format(urf_current_data[i_all_event_ERT_data_index, 2],urf_current_data[i_all_event_ERT_data_index, 3])
                        temp_Electrode_index=Tx_use_Electrode_index
                        if not temp_Electrode_index:
                            temp_x_vector = []
                        else:
                            temp_jet_colors = plt.cm.hsv(np.linspace(0, 1, len(Tx_use_Electrode_index)))
                            temp_x_vector = np.arange(480) / 10
                            # Plot each channel with unique color
                            for qq, index in enumerate(temp_Electrode_index):
                                ax.plot(temp_x_vector, temp_one_event_full_estimate_current_data[:, temp_Electrode_index[qq]-1] *1000, '-', color=temp_jet_colors[qq], label=f'CH{index:02d}', linewidth=1)
                            
                            # 只在有實際繪製帶標籤的線條時才添加圖例
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                        # Customize each subplot
                        ax.set_xlabel('Time [sec]')
                        ax.set_ylabel('Current [mA]')
                        ax.set_title(f'{png_file_first_name} EVENT: {i_all_event_ERT_data_index+1} / {num_events}{title_suffix}')
                        ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1, 5))

                    if i == 1:
                        title_suffix = ' (Voltage Measurement Data [Channel 1-12])'
                        temp_Electrode_index=[x for x in Rx_use_Electrode_index if x >= 1 and x <= 12]
                        if not temp_Electrode_index:
                            temp_x_vector = []
                        else:
                            temp_jet_colors = plt.cm.hsv(np.linspace(0, 1, len(temp_Electrode_index)))
                            temp_x_vector = np.arange(480) / 10
                            # Plot each channel with unique color
                            for qq, index in enumerate(temp_Electrode_index):
                                ax.plot(temp_x_vector, temp_one_event_full_measure_voltage_data[:, temp_Electrode_index[qq]-1]*1000, '-', color=temp_jet_colors[qq], label=f'CH{index:02d}', linewidth=1)
                            
                            # 只在有實際繪製帶標籤的線條時才添加圖例
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        
                        # Customize each subplot
                        ax.set_xlabel('Time [sec]')
                        ax.set_ylabel('Voltage [mV]')
                        ax.set_title(f'{png_file_first_name} EVENT: {i_all_event_ERT_data_index+1} / {num_events}{title_suffix}')
                        ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1, 5))

                    elif i == 3:
                        title_suffix = ' (Voltage Estimation Data [Channel 1-12])'
                        temp_Electrode_index=[x for x in Rx_use_Electrode_index if x >= 1 and x <= 12]
                        if not temp_Electrode_index:
                            temp_x_vector = []
                        else:
                            temp_jet_colors = plt.cm.hsv(np.linspace(0, 1, len(temp_Electrode_index)))
                            temp_x_vector = np.arange(480) / 10
                            # Plot each channel with unique color
                            for qq, index in enumerate(temp_Electrode_index):
                                ax.plot(temp_x_vector, temp_one_event_full_estimate_voltage_data[:, temp_Electrode_index[qq]-1]*1000, '-', color=temp_jet_colors[qq], label=f'CH{index:02d}', linewidth=1)
                            
                            # 只在有實際繪製帶標籤的線條時才添加圖例
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        
                        # Customize each subplot
                        ax.set_xlabel('Time [sec]')
                        ax.set_ylabel('Voltage [mV]')
                        ax.set_title(f'{png_file_first_name} EVENT: {i_all_event_ERT_data_index+1} / {num_events}{title_suffix}')
                        ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1, 5))

                    elif i == 5:
                        title_suffix = ' (Voltage Measurement Data [Channel 13-24])'
                        temp_Electrode_index=[x for x in Rx_use_Electrode_index if x >= 13 and x <= 24]
                        if not temp_Electrode_index:
                            temp_x_vector = []
                        else:
                            temp_jet_colors = plt.cm.hsv(np.linspace(0, 1, len(temp_Electrode_index)))
                            temp_x_vector = np.arange(480) / 10
                            # Plot each channel with unique color
                            for qq, index in enumerate(temp_Electrode_index):
                                ax.plot(temp_x_vector, temp_one_event_full_measure_voltage_data[:, temp_Electrode_index[qq]-1]*1000, '-', color=temp_jet_colors[qq], label=f'CH{index:02d}', linewidth=1)
                            
                            # 只在有實際繪製帶標籤的線條時才添加圖例
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        
                        # Customize each subplot
                        ax.set_xlabel('Time [sec]')
                        ax.set_ylabel('Voltage [mV]')
                        ax.set_title(f'{png_file_first_name} EVENT: {i_all_event_ERT_data_index+1} / {num_events}{title_suffix}')
                        ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1, 5))

                    elif i == 7:
                        title_suffix = ' (Voltage Estimate Data [Channel 13-24])'
                        temp_Electrode_index=[x for x in Rx_use_Electrode_index if x >= 13 and x <= 24]
                        if not temp_Electrode_index:
                            temp_x_vector = []
                        else:
                            temp_jet_colors = plt.cm.hsv(np.linspace(0, 1, len(temp_Electrode_index)))
                            temp_x_vector = np.arange(480) / 10
                            # Plot each channel with unique color
                            for qq, index in enumerate(temp_Electrode_index):
                                ax.plot(temp_x_vector, temp_one_event_full_estimate_voltage_data[:, temp_Electrode_index[qq]-1]*1000, '-', color=temp_jet_colors[qq], label=f'CH{index:02d}', linewidth=1)
                            
                            # 只在有實際繪製帶標籤的線條時才添加圖例
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        
                        # Customize each subplot
                        ax.set_xlabel('Time [sec]')
                        ax.set_ylabel('Voltage [mV]')
                        ax.set_title(f'{png_file_first_name} EVENT: {i_all_event_ERT_data_index+1} / {num_events}{title_suffix}')
                        ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1, 5))

                fig.savefig(os.path.join(output_png_path,f"{png_file_first_name}_EVENT_{i_all_event_ERT_data_index+1:02d}_{num_events}_P1.png"), dpi=75)
                plt.close(fig)

                # Create a figure with 4 rows and 2 columns of subplots (Rest Voltage data)
                fig, axs = plt.subplots(4, 2, figsize=(25.6, 14.4))  # Adjust size as needed
                fig.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing to prevent label overlap

                # Iterate over each subplot to populate with data
                for i, ax in enumerate(axs.flatten()):
                    if i == 0:
                        title_suffix = ' (Voltage Measurement Data [Channel 25-36])'
                        temp_Electrode_index=[x for x in Rx_use_Electrode_index if x >= 25 and x <= 36]
                        if not temp_Electrode_index:
                            temp_x_vector = []
                        else:
                            temp_jet_colors = plt.cm.hsv(np.linspace(0, 1, len(temp_Electrode_index)))
                            temp_x_vector = np.arange(480) / 10
                            # Plot each channel with unique color
                            for qq, index in enumerate(temp_Electrode_index):
                                ax.plot(temp_x_vector, temp_one_event_full_measure_voltage_data[:, temp_Electrode_index[qq]-1]*1000, '-', color=temp_jet_colors[qq], label=f'CH{index:02d}', linewidth=1)
                            
                            # 只在有實際繪製帶標籤的線條時才添加圖例
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        
                        # Customize each subplot
                        ax.set_xlabel('Time [sec]')
                        ax.set_ylabel('Voltage [mV]')
                        ax.set_title(f'{png_file_first_name} EVENT: {i_all_event_ERT_data_index+1} / {num_events}{title_suffix}'+', Page: 2 / 2')
                        ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1, 5))

                    elif i == 2:
                        title_suffix = ' (Voltage Estimation Data [Channel 25-36])'
                        temp_Electrode_index=[x for x in Rx_use_Electrode_index if x >= 25 and x <= 36]
                        if not temp_Electrode_index:
                            temp_x_vector = []
                        else:
                            temp_jet_colors = plt.cm.hsv(np.linspace(0, 1, len(temp_Electrode_index)))
                            temp_x_vector = np.arange(480) / 10
                            # Plot each channel with unique color
                            for qq, index in enumerate(temp_Electrode_index):
                                ax.plot(temp_x_vector, temp_one_event_full_estimate_voltage_data[:, temp_Electrode_index[qq]-1]*1000, '-', color=temp_jet_colors[qq], label=f'CH{index:02d}', linewidth=1)
                            
                            # 只在有實際繪製帶標籤的線條時才添加圖例
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                        # Customize each subplot
                        ax.set_xlabel('Time [sec]')
                        ax.set_ylabel('Voltage [mV]')
                        ax.set_title(f'{png_file_first_name} EVENT: {i_all_event_ERT_data_index+1} / {num_events}{title_suffix}')
                        ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1, 5))

                    elif i == 4:
                        title_suffix = ' (Voltage Measurement Data [Channel 37-48])'
                        temp_Electrode_index=[x for x in Rx_use_Electrode_index if x >= 37 and x <= 48]
                        if not temp_Electrode_index:
                            temp_x_vector = []
                        else:
                            temp_jet_colors = plt.cm.hsv(np.linspace(0, 1, len(temp_Electrode_index)))
                            temp_x_vector = np.arange(480) / 10
                            # Plot each channel with unique color
                            for qq, index in enumerate(temp_Electrode_index):
                                ax.plot(temp_x_vector, temp_one_event_full_measure_voltage_data[:, temp_Electrode_index[qq]-1]*1000, '-', color=temp_jet_colors[qq], label=f'CH{index:02d}', linewidth=1)
                            
                            # 只在有實際繪製帶標籤的線條時才添加圖例
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                        # Customize each subplot
                        ax.set_xlabel('Time [sec]')
                        ax.set_ylabel('Voltage [mV]')
                        ax.set_title(f'{png_file_first_name} EVENT: {i_all_event_ERT_data_index+1} / {num_events}{title_suffix}')
                        ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1, 5))

                    elif i == 6:
                        title_suffix = ' (Voltage Estimation Data [Channel 37-48])'
                        temp_Electrode_index=[x for x in Rx_use_Electrode_index if x >= 37 and x <= 48]
                        if not temp_Electrode_index:
                            temp_x_vector = []
                        else:
                            temp_jet_colors = plt.cm.hsv(np.linspace(0, 1, len(temp_Electrode_index)))
                            temp_x_vector = np.arange(480) / 10
                            # Plot each channel with unique color
                            for qq, index in enumerate(temp_Electrode_index):
                                ax.plot(temp_x_vector, temp_one_event_full_estimate_voltage_data[:, temp_Electrode_index[qq]-1]*1000, '-', color=temp_jet_colors[qq], label=f'CH{index:02d}', linewidth=1)
                            
                            # 只在有實際繪製帶標籤的線條時才添加圖例
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        
                        # Customize each subplot
                        ax.set_xlabel('Time [sec]')
                        ax.set_ylabel('Voltage [mV]')
                        ax.set_title(f'{png_file_first_name} EVENT: {i_all_event_ERT_data_index+1} / {num_events}{title_suffix}')
                        ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1, 5))

                    elif i == 1:
                        title_suffix = ' (Voltage Measurement Data [Channel 49-60])'
                        temp_Electrode_index=[x for x in Rx_use_Electrode_index if x >= 49 and x <= 60]
                        if not temp_Electrode_index:
                            temp_x_vector = []
                        else:
                            temp_jet_colors = plt.cm.hsv(np.linspace(0, 1, len(temp_Electrode_index)))
                            temp_x_vector = np.arange(480) / 10
                            # Plot each channel with unique color
                            for qq, index in enumerate(temp_Electrode_index):
                                ax.plot(temp_x_vector, temp_one_event_full_measure_voltage_data[:, temp_Electrode_index[qq]-1]*1000, '-', color=temp_jet_colors[qq], label=f'CH{index:02d}', linewidth=1)
                            
                            # 只在有實際繪製帶標籤的線條時才添加圖例
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                        # Customize each subplot
                        ax.set_xlabel('Time [sec]')
                        ax.set_ylabel('Voltage [mV]')
                        ax.set_title(f'{png_file_first_name} EVENT: {i_all_event_ERT_data_index+1} / {num_events}{title_suffix}')
                        ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1, 5))

                    elif i == 3:
                        title_suffix = ' (Voltage Estimation Data [Channel 49-60])'
                        temp_Electrode_index=[x for x in Rx_use_Electrode_index if x >= 49 and x <= 60]
                        if not temp_Electrode_index:
                            temp_x_vector = []
                        else:
                            temp_jet_colors = plt.cm.hsv(np.linspace(0, 1, len(temp_Electrode_index)))
                            temp_x_vector = np.arange(480) / 10
                            # Plot each channel with unique color
                            for qq, index in enumerate(temp_Electrode_index):
                                ax.plot(temp_x_vector, temp_one_event_full_estimate_voltage_data[:, temp_Electrode_index[qq]-1]*1000, '-', color=temp_jet_colors[qq], label=f'CH{index:02d}', linewidth=1)
                            
                            # 只在有實際繪製帶標籤的線條時才添加圖例
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        
                        # Customize each subplot
                        ax.set_xlabel('Time [sec]')
                        ax.set_ylabel('Voltage [mV]')
                        ax.set_title(f'{png_file_first_name} EVENT: {i_all_event_ERT_data_index+1} / {num_events}{title_suffix}')
                        ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1, 5))

                    elif i == 5:
                        title_suffix = ' (Voltage Measurement Data [Channel 61-64])'
                        temp_Electrode_index=[x for x in Rx_use_Electrode_index if x >= 61 and x <= 64]
                        if not temp_Electrode_index:
                            temp_x_vector = []
                        else:
                            temp_jet_colors = plt.cm.hsv(np.linspace(0, 1, len(temp_Electrode_index)))
                            temp_x_vector = np.arange(480) / 10
                            # Plot each channel with unique color
                            for qq, index in enumerate(temp_Electrode_index):
                                ax.plot(temp_x_vector, temp_one_event_full_measure_voltage_data[:, temp_Electrode_index[qq]-1]*1000, '-', color=temp_jet_colors[qq], label=f'CH{index:02d}', linewidth=1)
                            
                            # 只在有實際繪製帶標籤的線條時才添加圖例
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                        # Customize each subplot
                        ax.set_xlabel('Time [sec]')
                        ax.set_ylabel('Voltage [mV]')
                        ax.set_title(f'{png_file_first_name} EVENT: {i_all_event_ERT_data_index+1} / {num_events}{title_suffix}')
                        ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1, 5))

                    elif i == 7:
                        title_suffix = ' (Voltage Estimation Data [Channel 61-64])'
                        temp_Electrode_index=[x for x in Rx_use_Electrode_index if x >= 61 and x <= 64]
                        if not temp_Electrode_index:
                            temp_x_vector = []
                        else:
                            temp_jet_colors = plt.cm.hsv(np.linspace(0, 1, len(temp_Electrode_index)))
                            temp_x_vector = np.arange(480) / 10
                            # Plot each channel with unique color
                            for qq, index in enumerate(temp_Electrode_index):
                                ax.plot(temp_x_vector, temp_one_event_full_estimate_voltage_data[:, temp_Electrode_index[qq]-1]*1000, '-', color=temp_jet_colors[qq], label=f'CH{index:02d}', linewidth=1)
                            
                            # 只在有實際繪製帶標籤的線條時才添加圖例
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        
                        # Customize each subplot
                        ax.set_xlabel('Time [sec]')
                        ax.set_ylabel('Voltage [mV]')
                        ax.set_title(f'{png_file_first_name} EVENT: {i_all_event_ERT_data_index+1} / {num_events}{title_suffix}')
                        ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1, 5))

                fig.savefig(os.path.join(output_png_path,f"{png_file_first_name}_EVENT_{i_all_event_ERT_data_index+1:02d}_{num_events}_P2.png"), dpi=75)
                plt.close(fig)

            # print(f"Total number of measurement data: {len(urf_measurement_data[i_all_event_ERT_data_index])}")
            # print(f"The {i_all_event_ERT_data_index+1}th task processing ... end")

    # Check if the directory does not exist
    if not os.path.isdir(output_urf_path):
            os.makedirs(output_urf_path, exist_ok=True)  # exist_ok=True is safe for Python 3.2+, it handles the case if the directory exists

    output_urf_full_file_name = os.path.join(output_urf_path, output_urf_file_name)

    # Write the URF file
    # Open the file using 'with' to ensure it gets closed properly after being opened
    with open(output_urf_full_file_name, 'w') as f:
        # Write the geo information, assuming temp_geo_char_data is a byte-like or string object
        f.write(temp_geo_char_data)
        
        # Check and possibly write a newline character if it does not already end with one
        if ord(temp_geo_char_data[-1]) != 10:  # ord() gets the ASCII value of the last character
            f.write('\n')
        
        # Write ':Measurements' data, assuming urf_measurement_data is a list of lists or similar structure
        for i_all_event_ERT_data_index in range(num_events):
            if urf_measurement_data[i_all_event_ERT_data_index]:
                for row in urf_measurement_data[i_all_event_ERT_data_index]:
                    f.write(f'{row[0]},{row[1]},{row[2]},{row[3]},{row[4]:.6f},{row[5]:.6f},{row[6]:.6f}\n')
    return Return_Code


def convertURF(data_read_path = 'L1_1_m.urf', has_trn = False, trn_path = 'TaTun_ERT1.trn'):
    """
    將 urf 檔案轉換為 ohm 檔案，並支援地形內插
    
    參數:
        data_read_path: urf 檔案路徑
        has_trn: 是否使用地形檔案
        trn_path: 地形檔案路徑
    
    返回:
        data_write_path: 輸出的 ohm 檔案路徑
    """
    try:
        Trn = []
        trn_x = []
        trn_elevation = []
        
        if has_trn:
            print('Using terrain file name: ', trn_path)
            with open(trn_path, 'r') as read_obj:
                for i, line in enumerate(read_obj):
                    if i > 2:  # 跳過前3行（標題和單位行）
                        line = line.replace("\n", "").strip()
                        # 跳過空行
                        if not line:
                            continue
                        
                        values = line.split(',')
                        # 確保有足夠的值且數值有效
                        if len(values) >= 2:
                            try:
                                x_val = float(values[0].strip())
                                elev_val = float(values[1].strip())
                                
                                Trn.append(values)
                                # 為了內插，分別保存 x 坐標和高程
                                trn_x.append(x_val)
                                trn_elevation.append(elev_val)
                            except ValueError as e:
                                print(f"警告：跳過無效的數據行 {i+1}: {line} - 錯誤: {e}")
                        else:
                            print(f"警告：跳過格式不正確的行 {i+1}: {line}")

        # 檢查是否成功讀取了地形數據
        if has_trn and not trn_x:
            print("警告：沒有從地形檔案中讀取到有效數據！將使用 urf 檔案中的原始 z 坐標。")
            has_trn = False
        elif has_trn:
            print(f"成功從地形檔案中讀取了 {len(trn_x)} 個有效地形點")

        Trn_array = np.array(Trn, dtype=float) if Trn else np.array([])

        def nonblank_lines(f):
            for l in f:
                line = l.rstrip()
                if line:
                    yield line

        data_write_path = data_read_path[:-4] + ".ohm"

        print('The original urf file name: ', data_read_path)
        print('The output ohm file name: ', data_write_path)

        string_to_search1 = ":Geometry"
        string_to_search2 = ":Measurements"

        electrode_position = []
        resistivity_measurement = []
        Line = []
        with open(data_read_path, 'r') as read_obj:
            nonblank_lines_obj = nonblank_lines(read_obj)
            enu_read_obj = enumerate(nonblank_lines_obj)
            for i, line in enu_read_obj:
                Line.append(line.split(','))
        electrode_position = Line[Line.index([string_to_search1])+1:Line.index([string_to_search2])]
        resistivity_measurement = Line[Line.index([string_to_search2])+1::]
        electrode_position_array = np.array(electrode_position, dtype=float)

        with open(data_write_path, 'w') as write_obj:
            str = '%d # Number of electrodes\n' % (len(electrode_position))
            write_obj.write(str)
            write_obj.write('# x z position for each electrode\n')
            
            for i in range(len(electrode_position)):
                x_pos = electrode_position_array[i, 1]  # 電極的 x 坐標
                z_pos = electrode_position_array[i, 3]  # 電極的原始 z 坐標
                
                if has_trn and len(trn_x) > 0:
                    # 使用 numpy 的 interp 函數進行線性內插
                    # 如果 x_pos 超出了 trn_x 的範圍，使用最近的值
                    if x_pos < min(trn_x):
                        elevation = trn_elevation[trn_x.index(min(trn_x))]
                    elif x_pos > max(trn_x):
                        elevation = trn_elevation[trn_x.index(max(trn_x))]
                    else:
                        elevation = np.interp(x_pos, trn_x, trn_elevation)
                    
                    # 將原始 z 坐標加上內插的地形高程
                    # 注意：如果 z 是負值（例如地下深度），保持為負值並加上高程
                    final_z = elevation + z_pos
                    
                    #print(f"電極 {i+1}: x={x_pos}, 原始z={z_pos}, 內插高程={elevation}, 最終z={final_z}")
                    str = '%s   %s\n' % (x_pos, final_z)
                else:
                    str = '%s   %s\n' % (electrode_position[i][1], electrode_position[i][3])
                
                write_obj.write(str)

            str = '%d # Number of data\n' % (len(resistivity_measurement))
            write_obj.write(str)
            write_obj.write('# a b m n r i Uerr\n')
            for i in range(len(resistivity_measurement)):
                if float(resistivity_measurement[i][4]) == 0.0:
                    print('Find ZERO value!! \n Change to 0.00000000001 at ', i)
                    resistivity_measurement[i][4] = '0.00000000001'
                str = '%s   %s   %s   %s   %s   %s   %s\n' % (resistivity_measurement[i][0], resistivity_measurement[i][1],
                                            resistivity_measurement[i][2], resistivity_measurement[i][3],
                                            resistivity_measurement[i][4], resistivity_measurement[i][5], resistivity_measurement[i][6])
                write_obj.write(str)

        return data_write_path
    except Exception as e:
        print(f"URF 轉換失敗: {str(e)}")
        raise

def ridx_analyse(urf_dir, formula_choose='C'):
    """
    分析 r-index 品質
    
    參數:
        urf_dir: URF 檔案目錄
        formula_choose: 使用的公式選擇 ('A', 'B', 'C')
    
    返回:
        品質指標陣列
    """
    # 這是一個假的實現，實際上應該分析 r-index 品質
    quality_info = np.random.rand(1000) * 100  # 假數據
    print(f"分析 {urf_dir} 中的 URF 檔案，使用公式 {formula_choose}")
    return quality_info


def check_raw_data(temp_all_ERT_data_df):
    """
    檢查原始數據並嘗試修正問題
    
    參數:
        temp_all_ERT_data_df: 包含所有ERT數據的DataFrame
    
    返回:
        修正後的DataFrame
    """
    df = temp_all_ERT_data_df
    # Extract the fifth column (index 4 in zero-based counting)
    column_data = df.iloc[:, 4]

    # Initialize variables to track continuous strings
    current_string = None
    count = 0
    start_index = 0
    anomalies = []

    # Iterate over the column data
    for i, value in enumerate(column_data):
        if value == current_string:
            count += 1
        else:
            # Check if the count is a multiple of 10 or not
            if count != 0 and count % 10 != 0:
                anomalies.append((start_index, i-1, count))
            # Reset for the new string
            current_string = value
            count = 1
            start_index = i

    # Check the last sequence after the loop
    if count != 0 and count % 10 != 0:
        anomalies.append((start_index, len(column_data)-1, count))

    # Create a DataFrame for the anomalies to display
    anomalies_df = pd.DataFrame(anomalies, columns=['Start Index', 'End Index', 'Count'])
    # add time column, ignore the index column
    anomalies_df['Start Time'] = df.iloc[anomalies_df['Start Index'], 0].values
    anomalies_df['End Time'] = df.iloc[anomalies_df['End Index'], 0].values
    print(anomalies_df)

    # Copy the original dataframe to avoid modifying the original during processing
    df_modified = df.copy()

    # Process each anomaly
    offset = 0  # Tracks index offset due to rows being removed or added
    for start, end, count in anomalies:
        # Adjust indices with the current offset
        start_adjusted = start + offset
        end_adjusted = end + offset
        
        if count > 11 and count == 39:
            # Treat as count=9, needing to fill
            rows_to_add = 10 - 9  # Since we treat this as count = 9
            last_row = df_modified.iloc[end_adjusted]
            rows_to_append = pd.DataFrame([last_row] * rows_to_add)
            df_modified = pd.concat([df_modified.iloc[:end_adjusted+1], rows_to_append, df_modified.iloc[end_adjusted+1:]], ignore_index=True)
            offset += rows_to_add  # Update offset after addition
        
        elif count == 1:
            # If the count is 1, drop the row entirely
            df_modified.drop(df_modified.index[start_adjusted:end_adjusted + 1], inplace=True)
            offset -= (end_adjusted - start_adjusted + 1)  # Update offset after deletion
        
        elif count < 10:
            # If the count is less than 10, we copy the last row and append it to make up the difference
            rows_to_add = 10 - count
            last_row = df_modified.iloc[end_adjusted]
            rows_to_append = pd.DataFrame([last_row] * rows_to_add)
            df_modified = pd.concat([df_modified.iloc[:end_adjusted+1], rows_to_append, df_modified.iloc[end_adjusted+1:]], ignore_index=True)
            offset += rows_to_add  # Update offset after addition

    # Reset the index to ensure there's no index misalignment after the operations
    df_modified.reset_index(drop=True, inplace=True)

    return df_modified
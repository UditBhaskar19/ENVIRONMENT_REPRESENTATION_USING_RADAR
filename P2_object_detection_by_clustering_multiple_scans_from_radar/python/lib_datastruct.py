# =====================================================================================================================
# Author Name : Udit Bhaskar
# description : Data structures
# =====================================================================================================================

import numpy as np
import config

# =====================================================================================================================

def left_shift_history(data):
    data[:-1] = data[1:]
    return data


class meas_hist_buffer:
    """ A naive implementation of measurement hist buffer in case 
    we would like to store a sequence of radar frames """

    def __init__(self, history_size, max_num_meas, max_num_samples):
        self.top = -1
        self.history_size = history_size
        self.max_num_meas = max_num_meas + max_num_meas * max_num_samples

        self.meas_vector = np.zeros((self.history_size, self.max_num_meas, config.meas_dim), dtype=np.float32)    
        self.meas_covariance = np.zeros((self.history_size, self.max_num_meas, config.meas_dim, config.meas_dim), dtype=np.float32) 
        self.meas_status = np.zeros((self.history_size, self.max_num_meas, 2), dtype=np.bool8)                    # (dynamic-status, is_generated_sample)
        self.num_valid_meas = np.zeros((self.history_size, ), dtype=np.int32)                                     # number of valid meas
        self.timestamp = np.zeros((self.history_size, ), dtype=np.float32)                                        # timestamp
        self.sensorid = np.zeros((self.history_size, ), dtype=np.int32)                                           # sensorid


    def is_empty(self):
        return self.top == -1
    
    def is_full(self):
        return self.top == ( self.history_size - 1 )


    def update_buffer(
        self, 
        meas_vector, 
        meas_covariance,
        meas_status,
        num_meas, 
        timestamp,
        sensorid):

        if self.is_full():
            self.meas_vector = left_shift_history(self.meas_vector)
            self.meas_covariance = left_shift_history(self.meas_covariance)
            self.meas_status = left_shift_history(self.meas_status)
            self.num_valid_meas = left_shift_history(self.num_valid_meas)
            self.timestamp = left_shift_history(self.timestamp)
            self.sensorid = left_shift_history(self.sensorid)

        else: self.top += 1

        self.sensorid[self.top] = sensorid
        self.timestamp[self.top] = timestamp
        self.num_valid_meas[self.top] = num_meas
        if num_meas > 0:
            self.meas_vector[self.top, :num_meas] = meas_vector
            self.meas_covariance[self.top, :num_meas] = meas_covariance
            self.meas_status[self.top, :num_meas] = meas_status


    def get_buffer_size(self):
        return self.top + 1

    def get_num_meas_i(self, i):
        return self.num_valid_meas[i]

    def get_meas_vector_i(self, i):
        return self.meas_vector[i, :self.num_valid_meas[i]]

    def get_meas_covariance_i(self, i):
        return self.meas_covariance[i, :self.num_valid_meas[i]]

    def get_meas_status_i(self, i):
        return self.meas_status[i, :self.num_valid_meas[i], :]


    def get_all_valid_data(self):
        meas_vector = []
        meas_covariance = []
        meas_status = []
        num_valid_meas = []
        timestamp = []
        top_idx = []
        sensorid = []
        total_num_meas = 0

        if not self.is_empty():
            for i in range(self.top + 1):
                if self.num_valid_meas[i] > 0:
                    total_num_meas += self.num_valid_meas[i]
                    num_valid_meas.append(self.num_valid_meas[i])
                    sensorid.append(np.repeat(self.sensorid[i], self.num_valid_meas[i]))
                    timestamp.append(np.repeat(self.timestamp[i], self.num_valid_meas[i]))
                    top_idx.append(np.repeat(i, self.num_valid_meas[i]))
                    meas_vector.append(self.meas_vector[i, :self.num_valid_meas[i]])
                    meas_covariance.append(self.meas_covariance[i, :self.num_valid_meas[i]])
                    meas_status.append(self.meas_status[i, :self.num_valid_meas[i]])

        meas_vector = np.concatenate(meas_vector, axis=0)
        meas_covariance = np.concatenate(meas_covariance, axis=0)
        meas_status = np.concatenate(meas_status, axis=0)
        num_valid_meas = np.stack(num_valid_meas, axis=0)
        timestamp = np.concatenate(timestamp, axis=0)
        top_idx = np.concatenate(top_idx, axis=0)
        sensorid = np.concatenate(sensorid, axis=0)

        return (
            total_num_meas, 
            meas_vector, 
            meas_covariance,
            meas_status, 
            num_valid_meas, 
            timestamp,
            top_idx,
            sensorid )
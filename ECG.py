import neurokit2 as nk
import numpy as np
import pandas as pd
import time
import winsound  # For beep sound (Windows only)
import keyboard
import matplotlib.pyplot as plt
import tensorflow as tf  # Load trained ML model
from reportgenerator import ReportGenerator

class ECGSimulator:
    def __init__(self, model_path="ecg_cnn_model.h5"):
        self.sampling_rate = 125  # Hz
        self.chunk_size = 187  # Match the input dimension from your training (187 features)
        self.mode = "recorded"
        self.recorded_data = None
        self.r_peaks = []
        self.current_peak_index = 0
        self.alert_history = []
        self.csv_filename = "ecg_alerts.csv"
        self.report_generator = ReportGenerator()
        self.patient_data = {}
        self.full_analysis = []

    def set_patient_info(self, name, age, model_path="ecg_cnn_model.h5"):
        """Set patient information for report"""
        self.patient_data = {"name": name, "age": age}
        
        # Class labels for interpretable results
        self.class_labels = {
            0: "Normal",
            1: "Artial Premature",
            2: "Premature ventricular contraction",
            3: "Fusion of ventricular and normal",
            4: "Fusion of paced and normal"
        }

        # Load trained deep learning model
        self.model = tf.keras.models.load_model(model_path)
        print("✅ Model Loaded Successfully!")

    def load_recorded_data(self, file_path):
        """Load ECG data from CSV file and detect R-peaks"""
        try:
            df = pd.read_csv(file_path, header=None)
            df = df.iloc[:, :-1]
            self.recorded_data = df.values.flatten()  # Convert to 1D array

            # Detect R-peaks with CORRECT sampling rate (125 Hz)
            ecg_processed = nk.ecg_process(self.recorded_data, sampling_rate=self.sampling_rate)
            r_peaks_data = ecg_processed[1]['ECG_R_Peaks']
            if hasattr(r_peaks_data, 'values'):  # Check if it's a pandas object
                self.r_peaks = r_peaks_data.values
            else:  # It's already a NumPy array
                self.r_peaks = r_peaks_data
                
            print(f"Shape of recorded data after removing labels: {self.recorded_data.shape}")

            print(f"✅ Detected {len(self.r_peaks)} R-peaks in the ECG data.")

        except Exception as e:
            print(f"❌ Error loading ECG data: {e}")
            self.recorded_data = None

    def generate_ecg_chunk(self):
        """Fetch ECG segment centered around R-peaks"""
        if self.recorded_data is None or len(self.r_peaks) == 0:
            print("❌ No recorded ECG data or R-peaks detected.")
            return np.zeros(self.chunk_size)

        # Get the next R-peak index
        if self.current_peak_index >= len(self.r_peaks):
            self.current_peak_index = 0  # Restart when reaching the end

        r_peak = self.r_peaks[self.current_peak_index]
        self.current_peak_index += 1  # Move to next peak for the next chunk

        # Extract window centered on R-peak
        start = max(0, r_peak - self.chunk_size // 2)
        end = min(len(self.recorded_data), start + self.chunk_size)
        
        # Handle edge cases where we can't get enough samples
        if end - start < self.chunk_size:
            ecg_chunk = np.zeros(self.chunk_size)
            actual_data = self.recorded_data[start:end]
            ecg_chunk[:len(actual_data)] = actual_data
        else:
            ecg_chunk = self.recorded_data[start:end]

        return np.array(ecg_chunk)

    def detect_abnormality(self, ecg_chunk):
        """Use the trained model to detect abnormality (multi-class classification)"""
        try:
            # Normalize data similar to training
            mean = np.mean(ecg_chunk)
            std = np.std(ecg_chunk)
            if std == 0:
                std = 1  # Prevent division by zero

            ecg_chunk = (ecg_chunk - mean) / std
            
            # Reshape for the model input - flatten to 1D as per your model architecture
            ecg_chunk = ecg_chunk.reshape(1, -1)  # Reshape to match the model input
            
            # Get class probabilities
            prediction = self.model.predict(ecg_chunk, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Consider any non-normal class as abnormal
            is_abnormal = predicted_class != 0 and confidence > 0.6  # Apply confidence threshold
            
            return is_abnormal, confidence, predicted_class
        except Exception as e:
            print(f"❌ Error in detection: {e}")
            return False, 0, 0

    def trigger_alert(self, confidence, predicted_class):
        """Play beep sound and log the abnormality with class information"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        class_name = self.class_labels.get(predicted_class, "Unknown")
        
        alert_data = {
            "timestamp": timestamp, 
            "confidence": confidence,
            "class": predicted_class,
            "class_name": class_name
        }
        
        self.alert_history.append(alert_data)
        print(f"⚠️ ALERT: {class_name} detected at {timestamp} (Confidence: {confidence:.2f})")

        # Save alerts to CSV
        df = pd.DataFrame(self.alert_history)
        df.to_csv(self.csv_filename, index=False)

        # Beep sound (Windows only)
        winsound.Beep(1000, 500)

    def simulate_real_time(self):
        plt.ion()  # Enable interactive mode
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.set_title("Real-time ECG Signal")
        ax2.set_title("Abnormality Probability")
        
        # Initialize bar chart
        class_names = list(self.class_labels.values())
        bar_colors = ['green'] + ['red'] * (len(class_names) - 1)  # Normal is green, others red
        bars = ax2.bar(class_names, [0] * len(class_names), color=bar_colors)
        ax2.set_ylim(0, 1)
        
        print("Press 'q' to exit the simulation.")
        
        last_predictions = np.zeros(len(class_names))

        while True:
            if keyboard.is_pressed('q'):
                print("🛑 Simulation Stopped by User.")
                break

            ecg_chunk = self.generate_ecg_chunk()
            abnormal, confidence, predicted_class = self.detect_abnormality(ecg_chunk)

            # Update ECG plot
            ax1.clear()
            ax1.plot(ecg_chunk, label="ECG Signal", color='b')
            ax1.set_title("Real-time ECG Signal")
            ax1.legend()
            
            # Get full prediction array for visualization
            ecg_for_pred = ecg_chunk.reshape(1, -1)
            prediction_array = self.model.predict(ecg_for_pred, verbose=0)[0]
            
            # Smooth predictions for visualization (exponential moving average)
            alpha = 0.3  # Smoothing factor
            last_predictions = alpha * prediction_array + (1 - alpha) * last_predictions
            
            # Update bar heights
            for i, bar in enumerate(bars):
                bar.set_height(last_predictions[i])
            
            # Highlight the predicted class
            for i, bar in enumerate(bars):
                bar.set_color(bar_colors[i])  # Default color
                if i == predicted_class:
                    if predicted_class == 0:  # Normal class
                        bar.set_color('green')  # Ensure normal class is green
                    else:
                        bar.set_color('yellow')    # Highlight current prediction
            
            plt.tight_layout()
            plt.pause(0.1)  # Pause to update the plot

            if abnormal:
                # time.sleep(0.1)
                self.trigger_alert(confidence, predicted_class)

            time.sleep(0.5)  # Simulate real-time delay
            
        plt.ioff()  # Turn off interactive mode when done
        plt.show()
        self._generate_final_report()

    def _generate_final_report(self):
        """Generate final report after simulation"""
        # Calculate heart rate statistics
        rr_intervals = np.diff(self.r_peaks)
        heart_rate = 60 / (np.mean(rr_intervals) / self.sampling_rate)
        
        # Add sections to report
        self.report_generator.add_section(
            "ECG Analysis Summary",
            f"Average Heart Rate: {heart_rate:.1f} bpm\n"
            f"Total Abnormalities Detected: {len(self.alert_history)}\n"
            "Signal Quality: Good"
        )
        
        self.report_generator.add_section(
            "Technical Details",
            f"Analysis Duration: {len(self.r_peaks)/self.sampling_rate:.1f} seconds\n"
            f"Model Confidence Threshold: 0.6\n"
            f"Sampling Rate: {self.sampling_rate} Hz"
        )
        
        # Generate PDF report
        self.report_generator.generate_report(
            patient_data=self.patient_data,
            abnormalities=self.alert_history
        )

# ----------------- RUN SIMULATION -----------------
if __name__ == "__main__":
    
    simulator = ECGSimulator("ecg_cnn_model.h5")  # Load trained model
    simulator.set_patient_info("Maneet Gupta", 5)
    simulator.load_recorded_data(r"ptestdataset\ptbdb_abnormal.csv")
  # Load abnormal ECG dataset
    simulator.simulate_real_time()  # Run simulation
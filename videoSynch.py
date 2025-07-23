# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 16:30:49 2025

@author: Sasha Portnova

This is an audio-based video synchronization file for our FrnakenCap
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import subprocess
import librosa
import soundfile as sf
import tempfile
import ffmpeg

class VideoSynchronizer:
    def __init__(self, video_paths, reference_index=0):
        """
        Initialize the synchronizer with paths to the videos.
        
        Args:
            video_paths: List of paths to the video files
            reference_index: Index of the reference video (default: 0)
        """
        self.video_paths = video_paths
        self.audio_data = []
        self.sample_rates = []
        self.offsets = []
        self.reference_index = reference_index
        self.confidence_scores = []
        
    def extract_audio(self, output_dir=None):
        """
        Extract audio from all videos to WAV files using FFmpeg.
        
        Args:
            output_dir: Directory to store temporary audio files (if None, uses system temp dir)
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="video_sync_")
        elif not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        audio_paths = []
        
        for i, video_path in enumerate(self.video_paths):
            audio_path = os.path.join(output_dir, f"audio_{i}.wav")
            audio_paths.append(audio_path)
            
            # Use FFmpeg to extract audio with higher quality settings
            cmd = [
                "ffmpeg", "-i", video_path, 
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit little-endian format
                "-ar", "48000",  # Higher sample rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite without asking
                audio_path
            ]
            
            print(f"Extracting audio from {os.path.basename(video_path)}...")
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if result.returncode != 0:
                print(f"Error extracting audio: {result.stderr.decode()}")
                
        return audio_paths
    
    def load_audio_data(self, audio_paths):
        """
        Load audio data from extracted WAV files.
        
        Args:
            audio_paths: List of paths to audio files
        """
        self.audio_data = []
        self.sample_rates = []
        
        for i, audio_path in enumerate(audio_paths):
            try:
                # Read audio file
                audio_data, sample_rate = sf.read(audio_path)
                
                # Convert to mono if stereo
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
                
                # Normalize audio
                audio_data = audio_data.astype(np.float32)
                if np.max(np.abs(audio_data)) > 0:  # Avoid division by zero
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                self.audio_data.append(audio_data)
                self.sample_rates.append(sample_rate)
                
                print(f"Loaded audio from {os.path.basename(audio_path)}: "
                      f"Sample Rate = {sample_rate} Hz, "
                      f"Duration = {len(audio_data)/sample_rate:.2f} seconds")
            
            except Exception as e:
                print(f"Error loading audio file {audio_path}: {e}")
                # Add empty audio data as placeholder
                self.audio_data.append(np.zeros(1000))
                self.sample_rates.append(48000)
        
        print(f"Loaded {len(self.audio_data)} audio files")
        
    def preprocess_audio(self, sync_window_seconds=15):
        """
        Preprocess audio for better synchronization.
        - Normalize sample rates
        - Apply filtering
        - Compute audio energy envelope
        - Only process first N seconds for alignment
        
        Args:
            sync_window_seconds: Number of seconds to use for synchronization analysis
        """
        # Find the maximum sample rate for upsampling
        target_sr = max(self.sample_rates)
        
        # Normalize each audio track
        processed_audio = []
        for i, audio in enumerate(self.audio_data):
            print(f"Processing audio {i}: length={len(audio)}, sr={self.sample_rates[i]}")
            
            # Resample if needed
            if self.sample_rates[i] != target_sr:
                print(f"Resampling from {self.sample_rates[i]}Hz to {target_sr}Hz")
                audio = librosa.resample(
                    audio, 
                    orig_sr=self.sample_rates[i],
                    target_sr=target_sr
                )
            
            # LIMIT TO FIRST N SECONDS FOR SYNC ANALYSIS
            sync_samples = int(sync_window_seconds * target_sr)
            if len(audio) > sync_samples:
                audio = audio[:sync_samples]
                print(f"Limited audio to first {sync_window_seconds} seconds ({sync_samples} samples)")
            
            # Apply high-pass filter to remove low-frequency noise
            b, a = signal.butter(4, 100/(target_sr/2), 'highpass')
            audio = signal.filtfilt(b, a, audio)
            
            # Apply band-pass filter to focus on speech frequencies (300-3000 Hz)
            b, a = signal.butter(4, [300/(target_sr/2), 3000/(target_sr/2)], 'bandpass')
            audio = signal.filtfilt(b, a, audio)
            
            # Calculate frame parameters
            frame_length = int(0.025 * target_sr)  # 25ms frames
            hop_length = int(0.010 * target_sr)    # 10ms hop
            
            # Ensure frame_length is not larger than audio length
            if frame_length > len(audio):
                frame_length = len(audio) // 2
                hop_length = frame_length // 2
                print(f"Adjusted frame parameters: frame_length={frame_length}, hop_length={hop_length}")
            
            # Manual RMS calculation
            energy = np.array([np.sqrt(np.mean(audio[i:i+frame_length]**2)) 
                              for i in range(0, len(audio)-frame_length, hop_length)])
            
            # Normalize energy
            if np.max(energy) > 0:
                energy = energy / np.max(energy)
            
            processed_audio.append(energy)
            self.sample_rates[i] = target_sr
        
        self.audio_data = processed_audio
        self.hop_length = hop_length  # Save for time conversion
        print(f"All audio preprocessed and normalized to {target_sr}Hz")
        
    def compute_time_alignment(self):
        """
        Compute time offsets between videos using cross-correlation of energy envelopes.
        Fixed version with proper offset calculation and interpretation.
        
        Returns:
            List of time offsets in seconds relative to the reference video
        """
        if not self.audio_data:
            raise ValueError("Audio data must be loaded first")
        
        reference_audio = self.audio_data[self.reference_index]
        offsets = [0.0] * len(self.audio_data)
        confidence_scores = [1.0] * len(self.audio_data)
        
        print(f"Reference audio length: {len(reference_audio)} frames")
        
        for i, audio in enumerate(self.audio_data):
            if i == self.reference_index:
                continue  # Skip the reference video
            
            print(f"\nAligning video {i} with reference...")
            print(f"Video {i} audio length: {len(audio)} frames")
            
            # Compute cross-correlation
            correlation = signal.correlate(reference_audio, audio, mode='full')
            
            # Normalize correlation
            norm_factor = np.sqrt(np.sum(reference_audio**2) * np.sum(audio**2))
            if norm_factor > 0:
                correlation = correlation / norm_factor
            
            # Find the peak correlation
            max_idx = np.argmax(correlation)
            max_correlation = correlation[max_idx]
            
            print(f"Max correlation: {max_correlation:.3f} at index {max_idx}")
            
            # Calculate confidence score
            sharpness = max_correlation / (np.median(np.abs(correlation)) + 1e-10)
            confidence = max_correlation * min(1.0, sharpness/5)
            confidence_scores[i] = confidence
            
            # Convert correlation index to time shift
            # The correlation result has length len(reference_audio) + len(audio) - 1
            # The center index (zero lag) is at position len(audio) - 1
            zero_lag_idx = len(audio) - 1
            lag_samples = max_idx - zero_lag_idx
            
            # Convert lag from samples to time
            # Positive lag means audio[i] is delayed relative to reference
            # Negative lag means audio[i] leads the reference
            time_shift = lag_samples * self.hop_length / self.sample_rates[i]
            
            offsets[i] = time_shift
            
            print(f"Lag in samples: {lag_samples}")
            print(f"Time shift: {time_shift:.3f} seconds")
            print(f"Confidence: {confidence:.3f}")
            
            # Interpretation:
            if time_shift > 0:
                print(f"  → Video {i} starts {time_shift:.3f}s AFTER reference")
            elif time_shift < 0:
                print(f"  → Video {i} starts {abs(time_shift):.3f}s BEFORE reference")
            else:
                print(f"  → Video {i} is perfectly aligned with reference")
        
        self.offsets = offsets
        self.confidence_scores = confidence_scores
        
        # Print summary
        print(f"\n=== ALIGNMENT SUMMARY ===")
        for i, offset in enumerate(offsets):
            if i == self.reference_index:
                print(f"Video {i}: REFERENCE (0.000s)")
            else:
                direction = "AFTER" if offset > 0 else "BEFORE"
                print(f"Video {i}: {abs(offset):.3f}s {direction} reference (confidence: {confidence_scores[i]:.3f})")
        
        return offsets
    
    def refine_alignment_with_full_audio(self, window_size=5.0):
        """
        Refine alignment using full audio around the initial offset estimate.
        
        Args:
            window_size: Size of the audio window in seconds around initial offset
        """
        print("Refining alignment with full audio...")
        
        # Load original audio again for refinement
        audio_paths = self.extract_audio()
        original_audio = []
        original_sr = []
        
        for audio_path in audio_paths:
            audio, sr = sf.read(audio_path)
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            original_audio.append(audio)
            original_sr.append(sr)
        
        refined_offsets = self.offsets.copy()
        
        # Get reference audio
        ref_audio = original_audio[self.reference_index]
        ref_sr = original_sr[self.reference_index]
        
        for i in range(len(original_audio)):
            if i == self.reference_index:
                continue
                
            audio = original_audio[i]
            sr = original_sr[i]
            
            # Calculate window in samples
            window_samples = int(window_size * sr)
            
            # Convert initial offset to samples
            initial_offset_samples = int(self.offsets[i] * sr)
            
            # Create window around initial estimate
            if initial_offset_samples > 0:
                # Audio starts after reference
                ref_start = initial_offset_samples - window_samples
                ref_start = max(0, ref_start)
                ref_end = min(len(ref_audio), initial_offset_samples + window_samples)
                ref_window = ref_audio[ref_start:ref_end]
                
                audio_start = 0
                audio_end = min(len(audio), 2 * window_samples)
                audio_window = audio[audio_start:audio_end]
            else:
                # Audio starts before reference
                ref_start = 0
                ref_end = min(len(ref_audio), 2 * window_samples)
                ref_window = ref_audio[ref_start:ref_end]
                
                audio_start = -initial_offset_samples - window_samples
                audio_start = max(0, audio_start)
                audio_end = min(len(audio), -initial_offset_samples + window_samples)
                audio_window = audio[audio_start:audio_end]
            
            # Ensure we have enough data
            if len(ref_window) < window_samples or len(audio_window) < window_samples:
                print(f"Not enough data for refinement of video {i}, keeping initial estimate")
                continue
                
            # Compute cross-correlation
            correlation = signal.correlate(ref_window, audio_window, mode='full')
            max_idx = np.argmax(correlation)
            
            # Convert to time offset
            shift_idx = max_idx - (len(ref_window) + len(audio_window) - 1) // 2
            
            # Adjust for window position
            if initial_offset_samples > 0:
                refined_shift = (ref_start + shift_idx) / sr
            else:
                refined_shift = (shift_idx - audio_start) / sr
            
            # Update offset with refined value
            refined_offsets[i] = refined_shift
            print(f"Refined: Video {i} is shifted by {refined_shift:.3f} seconds (initial: {self.offsets[i]:.3f}s)")
        
        self.offsets = refined_offsets
        
        # Clean up temporary files
        for path in audio_paths:
            if os.path.exists(path):
                os.remove(path)
                
        return refined_offsets
    
    def visualize_alignment(self, time_window=10, show_original=True):
        """
        Enhanced visualization with better labeling and debugging info.
        """
        if show_original:
            # Need to reload original audio
            audio_paths = self.extract_audio()
            audio_data = []
            sample_rates = []
            
            for audio_path in audio_paths:
                audio, sr = sf.read(audio_path)
                # Convert to mono if stereo
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                # Limit to same window used for sync analysis
                sync_samples = int(15 * sr)  # Match the sync window
                if len(audio) > sync_samples:
                    audio = audio[:sync_samples]
                audio_data.append(audio)
                sample_rates.append(sr)
        else:
            audio_data = self.audio_data
            sample_rates = self.sample_rates
            
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # Plot 1: Original alignment
        ax1.set_title("BEFORE Alignment", fontsize=14, fontweight='bold')
        for i, audio in enumerate(audio_data):
            sr = sample_rates[i]
            t = np.arange(len(audio)) / sr
            mask = t < time_window
            
            color = colors[i % len(colors)]
            ax1.plot(t[mask], audio[mask], label=f"Video {i}", color=color, alpha=0.7)
            
        ax1.set_ylabel("Amplitude")
        ax1.set_xlabel("Time (s)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: After alignment (with offsets applied)
        ax2.set_title("AFTER Alignment", fontsize=14, fontweight='bold')
        for i, audio in enumerate(audio_data):
            sr = sample_rates[i]
            t = np.arange(len(audio)) / sr
            
            # Apply the calculated offset
            t_shifted = t + self.offsets[i]
            
            mask = (t_shifted >= 0) & (t_shifted < time_window)
            
            color = colors[i % len(colors)]
            label = f"Video {i} (offset: {self.offsets[i]:+.3f}s)"
            ax2.plot(t_shifted[mask], audio[mask], label=label, color=color, alpha=0.7)
            
        ax2.set_ylabel("Amplitude")
        ax2.set_xlabel("Time (s)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cross-correlation visualization
        if len(audio_data) >= 2:
            ax3.set_title("Cross-correlation (Video 1 vs Reference)", fontsize=14, fontweight='bold')
            
            ref_audio = audio_data[self.reference_index]
            other_audio = audio_data[1 if self.reference_index != 1 else 0]
            
            # Compute and plot cross-correlation
            correlation = signal.correlate(ref_audio, other_audio, mode='full')
            
            # Create time axis for correlation
            zero_lag_idx = len(other_audio) - 1
            lag_samples = np.arange(len(correlation)) - zero_lag_idx
            lag_time = lag_samples * (1.0 / sample_rates[0])  # Assuming same sample rate
            
            ax3.plot(lag_time, correlation, 'purple', alpha=0.7)
            
            # Mark the maximum
            max_idx = np.argmax(correlation)
            max_lag = lag_time[max_idx]
            ax3.axvline(x=max_lag, color='red', linestyle='--', 
                       label=f'Max correlation at {max_lag:.3f}s')
            ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3, label='Zero lag')
            
            ax3.set_xlabel("Lag (s)")
            ax3.set_ylabel("Cross-correlation")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Limit x-axis for better visualization
            ax3.set_xlim([-5, 5])
        
        plt.tight_layout()
        plt.show()
        
        # Clean up if we reloaded original audio
        if show_original:
            for path in audio_paths:
                if os.path.exists(path):
                    os.remove(path)
    
    def create_synchronized_videos(self, output_prefix="synced_"):
        """
        Create synchronized videos where BOTH videos are trimmed to have the same final duration.
        Uses frame-exact synchronization to ensure identical frame counts.
        
        Args:
            output_prefix: Prefix for output filenames
        """
        
        def get_fps(video_path):
            """Get frame rate from video"""
            try:
                probe = ffmpeg.probe(video_path, select_streams='v:0')
                video_stream = probe['streams'][0]
                fps_str = video_stream['r_frame_rate']
                fps = eval(fps_str)  # Handle fractions like "30000/1001"
                return fps
            except Exception as e:
                print(f"Error getting fps for {video_path}: {e}")
                return 30.0  # fallback
        
        # Get duration of each video using ffprobe
        durations = []
        fps_values = []
        for video_path in self.video_paths:
            cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            duration = float(result.stdout.strip())
            durations.append(duration)
            
            fps = get_fps(video_path)
            fps_values.append(fps)
        
        # Use the first video's fps as reference (or set a fixed fps like 30)
        reference_fps = fps_values[0]
        print(f"Using reference frame rate: {reference_fps:.2f} fps")
        
        print(f"Original video durations: {durations}")
        print(f"Original video fps: {[f'{fps:.2f}' for fps in fps_values]}")
        print(f"Computed offsets: {self.offsets}")
        
        # Calculate the effective start and end times for each video
        video_specs = []
        
        for i in range(len(self.video_paths)):
            offset = self.offsets[i]
            original_duration = durations[i]
            
            # Calculate when this video effectively starts and ends in the synchronized timeline
            if offset >= 0:
                # This video starts AFTER the reference point
                effective_start = offset  # Video starts this many seconds into the timeline
                effective_end = offset + original_duration  # Video ends here
                trim_start = 0  # Don't trim the beginning of this video
            else:
                # This video starts BEFORE the reference point
                effective_start = 0  # Video content starts at timeline zero
                effective_end = original_duration + offset  # Video ends earlier due to offset
                trim_start = abs(offset)  # Trim this much from the beginning
            
            video_specs.append({
                'index': i,
                'offset': offset,
                'original_duration': original_duration,
                'effective_start': effective_start,
                'effective_end': effective_end,
                'trim_start': trim_start,
                'fps': fps_values[i]
            })
        
        # Find the common timeline boundaries
        # All videos should start at the latest effective start time
        common_start = max(spec['effective_start'] for spec in video_specs)
        # All videos should end at the earliest effective end time
        common_end = min(spec['effective_end'] for spec in video_specs)
        
        # Calculate the final synchronized duration
        final_duration = common_end - common_start
        
        if final_duration <= 0:
            print("ERROR: No overlapping time between videos after alignment!")
            return
        
        # Convert to exact frames to ensure identical frame counts
        final_frame_count = round(final_duration * reference_fps)
        exact_final_duration = final_frame_count / reference_fps
        
        print(f"Common timeline: {common_start:.2f}s to {common_end:.2f}s")
        print(f"Final synchronized duration: {final_duration:.2f}s -> {exact_final_duration:.2f}s")
        print(f"Target frame count: {final_frame_count} frames")
        
        # Process each video
        for spec in video_specs:
            i = spec['index']
            video_path = self.video_paths[i]
            
            # Calculate trim parameters for this specific video
            # How much to trim from the start (original trim + adjustment to common start)
            start_trim = spec['trim_start'] + (common_start - spec['effective_start'])
            
            # Convert to exact frames for frame-precise trimming
            start_frame = round(start_trim * reference_fps)
            start_trim_exact = start_frame / reference_fps
            
            # Calculate exact end time based on frame count
            end_time_exact = start_trim_exact + exact_final_duration
            
            # Get the directory and filename of the original video
            directory = os.path.dirname(video_path)
            filename = os.path.basename(video_path)
            output_path = os.path.join(directory, f"{output_prefix}{filename}")
            
            print(f"\nProcessing video {i}: {filename}")
            print(f"  Original duration: {spec['original_duration']:.2f}s")
            print(f"  Offset: {spec['offset']:.2f}s")
            print(f"  Trim start: {start_trim:.4f}s -> {start_trim_exact:.4f}s (frame {start_frame})")
            print(f"  End time: {end_time_exact:.4f}s")
            print(f"  Final duration: {exact_final_duration:.4f}s ({final_frame_count} frames)")
            
            try:
                # Use frame-based selection for exact synchronization
                stream = ffmpeg.input(video_path)
                
                # Select specific frame range
                end_frame = start_frame + final_frame_count
                stream = ffmpeg.filter(
                    stream, 
                    'select', 
                    f'gte(n,{start_frame})*lt(n,{end_frame})'
                )
                
                # Reset presentation timestamps
                stream = ffmpeg.filter(stream, 'setpts', 'N/FRAME_RATE/TB')
                
                # Output with exact frame rate
                stream = ffmpeg.output(
                    stream, 
                    output_path,
                    vcodec='libx264',
                    preset='fast',
                    crf=17,
                    acodec='aac',
                    audio_bitrate='192k',
                    r=reference_fps,
                    vsync='cfr'
                )
                
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                
                # Verify the output duration and frame count
                try:
                    probe = ffmpeg.probe(output_path)
                    actual_duration = float(probe['format']['duration'])
                    
                    # Try to get frame count
                    try:
                        video_stream = probe['streams'][0]
                        if 'nb_frames' in video_stream:
                            actual_frames = int(video_stream['nb_frames'])
                            print(f"  ✓ Created: {output_path}")
                            print(f"    Actual: {actual_duration:.4f}s, {actual_frames} frames")
                            if actual_frames != final_frame_count:
                                print(f"    WARNING: Expected {final_frame_count} frames, got {actual_frames}")
                        else:
                            estimated_frames = round(actual_duration * reference_fps)
                            print(f"  ✓ Created: {output_path}")
                            print(f"    Actual: {actual_duration:.4f}s, ~{estimated_frames} frames (estimated)")
                    except:
                        print(f"  ✓ Created: {output_path} (actual: {actual_duration:.4f}s)")
                        
                except ffmpeg.Error:
                    print(f"  ✓ Created: {output_path} (duration verification failed)")
            
            except ffmpeg.Error as e:
                print(f"  ✗ Error creating synchronized video: {e}")
        
        print(f"\nSynchronization complete! All videos should have exactly {final_frame_count} frames ({exact_final_duration:.4f}s).")
    
    def run_synchronization(self, visualize=True, sync_window_seconds=15):
        """
        Run the complete synchronization pipeline
        
        Args:
            visualize: Whether to visualize the results
            sync_window_seconds: Number of seconds to analyze for synchronization
            
        Returns:
            List of time offsets in seconds relative to reference video
        """
        print("Starting video synchronization...")
        print(f"Using first {sync_window_seconds} seconds for alignment analysis...")
        
        # Extract and load audio
        audio_paths = self.extract_audio()
        self.load_audio_data(audio_paths)
        
        # Preprocess audio for better sync (LIMITED TO FIRST N SECONDS)
        self.preprocess_audio(sync_window_seconds=sync_window_seconds)
        
        # Compute initial alignment
        self.compute_time_alignment()
        
        # Refine alignment using full audio (still limited to the window)
        # Note: You might want to modify refine_alignment_with_full_audio to also use the window
        # self.refine_alignment_with_full_audio()
        
        # Visualize if requested
        if visualize:
            self.visualize_alignment()
        
        # Create synchronized videos with equal lengths
        self.create_synchronized_videos()
        
        print("Synchronization complete!")
        
        # Clean up temporary files
        for path in audio_paths:
            if os.path.exists(path):
                os.remove(path)
        
        return self.offsets

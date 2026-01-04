"""
Smart Attendance Management System v2.0
========================================
Enterprise-grade face recognition attendance system.

Features:
- Multiple Face Detection (simultaneous recognition)
- Excel Export with Charts
- Modern CustomTkinter GUI
- Dark/Light Theme Toggle
- Secure Password Hashing (SHA-256)

Author: Karthigaiselvam R
License: MIT
"""

import customtkinter as ctk
import cv2
import csv
import os
import json
import hashlib
import secrets
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
from datetime import datetime
import time
import threading
from tkinter import messagebox
from typing import Optional, Dict, List, Tuple
from excel_exporter import export_to_excel


# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================
class ConfigManager:
    """Manages application configuration and secure password hashing."""
    
    CONFIG_FILE = "config.json"
    DEFAULT_CONFIG = {
        "admin": {"password_hash": "", "salt": ""},
        "theme": "dark",
        "camera_id": 0,
        "recognition_threshold": 70,
        "capture_duration": 20,
        "images_per_student": 70
    }
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return self.DEFAULT_CONFIG.copy()
        return self.DEFAULT_CONFIG.copy()
    
    def save_config(self):
        with open(self.CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    @staticmethod
    def hash_password(password: str, salt: str = None) -> Tuple[str, str]:
        """Hash password using SHA-256 with salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.sha256((password + salt).encode()).hexdigest()
        return hashed, salt
    
    def verify_password(self, password: str) -> bool:
        """Verify password against stored hash."""
        stored_hash = self.config["admin"].get("password_hash", "")
        salt = self.config["admin"].get("salt", "")
        if not stored_hash or not salt:
            return password == "admin123"
        check_hash, _ = self.hash_password(password, salt)
        return check_hash == stored_hash
    
    def set_password(self, new_password: str):
        """Set new admin password with secure hashing."""
        hashed, salt = self.hash_password(new_password)
        self.config["admin"]["password_hash"] = hashed
        self.config["admin"]["salt"] = salt
        self.save_config()
    
    def get_theme(self) -> str:
        return self.config.get("theme", "dark")
    
    def set_theme(self, theme: str):
        self.config["theme"] = theme
        self.save_config()


# ============================================================================
# FACE RECOGNITION ENGINE
# ============================================================================
class FaceRecognitionEngine:
    """Production-grade face detection and recognition engine."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.recognizer = None
        self.face_cascade = None
        self.model_path = "TrainingImageLabel/Trainner.yml"
        self.student_df: Optional[pd.DataFrame] = None
        self.is_model_loaded = False
        self._initialize_cascade()
    
    def _initialize_cascade(self):
        """Initialize Haar Cascade classifier."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            # Fallback to local file
            local_cascade = "haarcascade_frontalface_default.xml"
            if os.path.exists(local_cascade):
                self.face_cascade = cv2.CascadeClassifier(local_cascade)
    
    def load_model(self) -> bool:
        """Load trained LBPH model and student data."""
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            if os.path.exists(self.model_path):
                self.recognizer.read(self.model_path)
                self.is_model_loaded = True
            
            student_csv = "StudentDetails/StudentDetails.csv"
            if os.path.exists(student_csv):
                self.student_df = pd.read_csv(student_csv)
            
            return self.is_model_loaded
        except Exception as e:
            print(f"[ERROR] Model loading failed: {e}")
            return False
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect and recognize multiple faces in frame."""
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )
        
        detections = []
        threshold = self.config.config.get("recognition_threshold", 70)
        
        for (x, y, w, h) in faces:
            detection = {
                "id": None, "name": "Unknown", "confidence": 0,
                "bbox": (x, y, w, h), "recognized": False
            }
            
            if self.is_model_loaded and self.recognizer is not None:
                try:
                    face_roi = gray[y:y+h, x:x+w]
                    id_pred, confidence = self.recognizer.predict(face_roi)
                    
                    if confidence < threshold and self.student_df is not None:
                        matches = self.student_df.loc[
                            self.student_df['Enrollment'] == id_pred
                        ]['Name'].values
                        if len(matches) > 0:
                            detection.update({
                                "id": id_pred, "name": str(matches[0]),
                                "confidence": confidence, "recognized": True
                            })
                except Exception:
                    pass
            
            detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        for det in detections:
            x, y, w, h = det["bbox"]
            color = (0, 200, 0) if det["recognized"] else (0, 0, 200)
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Label with background
            label = det["name"]
            if det["id"]:
                label = f"{det['name']} [{det['id']}]"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(frame, (x, y-th-10), (x+tw+10, y), color, -1)
            cv2.putText(frame, label, (x+5, y-5), font, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def train_model(self) -> Tuple[bool, str]:
        """Train LBPH face recognition model."""
        if self.face_cascade is None:
            return False, "Face cascade not initialized"
        
        image_dir = "TrainingImage"
        if not os.path.exists(image_dir) or not os.listdir(image_dir):
            return False, "No training images found"
        
        face_samples = []
        ids = []
        
        for image_name in os.listdir(image_dir):
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            image_path = os.path.join(image_dir, image_name)
            try:
                pil_img = Image.open(image_path).convert('L')
                img_np = np.array(pil_img, 'uint8')
                
                parts = os.path.splitext(image_name)[0].split('.')
                if len(parts) >= 2:
                    id_val = int(parts[1])
                    faces = self.face_cascade.detectMultiScale(img_np)
                    for (x, y, w, h) in faces:
                        face_samples.append(img_np[y:y+h, x:x+w])
                        ids.append(id_val)
            except Exception:
                continue
        
        if not face_samples:
            return False, "No valid faces found in training images"
        
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.train(face_samples, np.array(ids))
        
        os.makedirs("TrainingImageLabel", exist_ok=True)
        self.recognizer.save(self.model_path)
        self.is_model_loaded = True
        
        return True, f"Model trained successfully with {len(face_samples)} face samples"


# ============================================================================
# MAIN APPLICATION
# ============================================================================
class SmartAttendanceApp(ctk.CTk):
    """Enterprise Smart Attendance Application with Modern UI."""
    
    # Color scheme
    ACCENT_COLOR = "#1E88E5"
    SUCCESS_COLOR = "#43A047"
    ERROR_COLOR = "#E53935"
    
    def __init__(self):
        super().__init__()
        
        # Initialize core components
        self.config_manager = ConfigManager()
        self.face_engine = FaceRecognitionEngine(self.config_manager)
        self.face_engine.load_model()
        
        # Apply theme
        theme = self.config_manager.get_theme()
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")
        
        # Window configuration
        self.title("Smart Attendance System")
        self.geometry("1400x800")
        self.minsize(1200, 700)
        
        # Load Icons
        self.icons = self._load_icons()
        
        # Session state
        self.attendance_session = pd.DataFrame(columns=['Enrollment', 'Name', 'Date', 'Time'])
        self.current_subject = ""
        
        # Build interface
        self._create_sidebar()
        self._create_main_container()
        self._show_home()
        
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _load_icons(self):
        """Load and resize icons."""
        icons = {}
        icon_path = "assets"
        try:
            icons["logo"] = ctk.CTkImage(Image.open(os.path.join(icon_path, "logo.png")), size=(40, 40))
            # Adjusted size: Increased as requested (290x290)
            icons["logo_large"] = ctk.CTkImage(Image.open(os.path.join(icon_path, "logo.png")), size=(290, 290))
            icons["home"] = ctk.CTkImage(Image.open(os.path.join(icon_path, "home.png")), size=(24, 24))
            icons["capture"] = ctk.CTkImage(Image.open(os.path.join(icon_path, "camera.png")), size=(24, 24))
            icons["train"] = ctk.CTkImage(Image.open(os.path.join(icon_path, "train.png")), size=(24, 24))
            icons["attendance"] = ctk.CTkImage(Image.open(os.path.join(icon_path, "attendance.png")), size=(24, 24))
            icons["records"] = ctk.CTkImage(Image.open(os.path.join(icon_path, "records.png")), size=(24, 24))
            icons["admin"] = ctk.CTkImage(Image.open(os.path.join(icon_path, "admin.png")), size=(24, 24))
        except Exception as e:
            print(f"Error loading icons: {e}")
            return None
        return icons

    def _create_sidebar(self):
        """Create navigation sidebar."""
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color=("#E3E3E3", "#1A1A1A"))
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        
        # Branding
        brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand_frame.pack(fill="x", pady=(30, 20), padx=20)
        
        if self.icons and "logo" in self.icons:
             ctk.CTkLabel(brand_frame, text="", image=self.icons["logo"]).pack(side="left")
        
        text_frame = ctk.CTkFrame(brand_frame, fg_color="transparent")
        text_frame.pack(side="left", padx=10)
        
        ctk.CTkLabel(
            text_frame, text="SMART", font=ctk.CTkFont(size=20, weight="bold"),
            text_color=self.ACCENT_COLOR
        ).pack(anchor="w")
        ctk.CTkLabel(
            text_frame, text="ATTENDANCE", font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w")
        
        # Divider
        ctk.CTkFrame(self.sidebar, height=2, fg_color=("#CCCCCC", "#333333")).pack(fill="x", padx=20, pady=(0, 20))
        
        # Navigation items
        nav_items = [
            ("Home", self._show_home, "home"),
            ("Capture Images", self._show_capture, "capture"),
            ("Train Model", self._on_train_model, "train"),
            ("Take Attendance", self._show_attendance, "attendance"),
            ("View Records", self._show_records, "records"),
            ("Admin Panel", self._show_admin_login, "admin"),
        ]
        
        for text, command, icon_key in nav_items:
            icon = self.icons.get(icon_key) if self.icons else None
            btn = ctk.CTkButton(
                self.sidebar, text=text, command=command,
                width=200, height=42, corner_radius=8,
                font=ctk.CTkFont(size=14),
                fg_color="transparent", text_color=("gray10", "gray90"),
                hover_color=("#D0D0D0", "#2A2A2A"),
                anchor="w", image=icon, compound="left"
            )
            btn.pack(pady=4, padx=20)
        
        # Spacer
        ctk.CTkFrame(self.sidebar, fg_color="transparent").pack(expand=True)
        
        # Theme toggle
        theme_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        theme_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(theme_frame, text="Dark Mode", font=ctk.CTkFont(size=12)).pack(side="left")
        self.theme_switch = ctk.CTkSwitch(
            theme_frame, text="", width=50, command=self._toggle_theme
        )
        self.theme_switch.pack(side="right")
        if self.config_manager.get_theme() == "dark":
            self.theme_switch.select()
        
        # Version info
        ctk.CTkLabel(
            self.sidebar, text="v2.0.0", font=ctk.CTkFont(size=10),
            text_color="gray50"
        ).pack(pady=(5, 20))
    
    def _create_main_container(self):
        """Create main content container."""
        self.main_container = ctk.CTkFrame(self, corner_radius=0, fg_color=("#F5F5F5", "#121212"))
        self.main_container.pack(side="right", fill="both", expand=True)
        
        # Content frames
        self.frames = {
            "home": ctk.CTkFrame(self.main_container, fg_color="transparent"),
            "capture": ctk.CTkFrame(self.main_container, fg_color="transparent"),
            "attendance": ctk.CTkFrame(self.main_container, fg_color="transparent"),
            "records": ctk.CTkFrame(self.main_container, fg_color="transparent"),
        }
    
    def _hide_all_frames(self):
        for frame in self.frames.values():
            frame.pack_forget()
    
    def _clear_frame(self, frame: ctk.CTkFrame):
        for widget in frame.winfo_children():
            widget.destroy()
    
    # ========================================================================
    # HOME VIEW with ANIMATION
    # ========================================================================
    def _show_home(self):
        self._hide_all_frames()
        self._stop_animation()  # Stop any existing animation
        
        frame = self.frames["home"]
        self._clear_frame(frame)
        frame.pack(fill="both", expand=True) 
        
        # Main Container (Standard Frame)
        main_layout = ctk.CTkFrame(frame, fg_color="transparent")
        main_layout.pack(fill="both", expand=True, padx=20, pady=5) # Reduced main padding to 5
        
        # Header with branding
        header_frame = ctk.CTkFrame(main_layout, fg_color="transparent")
        header_frame.pack(pady=(0, 0)) # Minimized padding
        
        # Logo Container for Animation - Increased size
        self.logo_container = ctk.CTkFrame(header_frame, fg_color="transparent", height=310, width=310)
        self.logo_container.pack(pady=5)
        self.logo_container.pack_propagate(False) # Prevent shrinking
        
        if self.icons and "logo_large" in self.icons:
            # Note: Parent is now self.logo_container
            self.home_logo_label = ctk.CTkLabel(self.logo_container, text="", image=self.icons["logo_large"])
            self.home_logo_label.place(relx=0.5, rely=0.5, anchor="center")
            self._start_animation()

        ctk.CTkLabel(
            header_frame, text="Welcome to Smart Attendance",
            font=ctk.CTkFont(size=32, weight="bold")
        ).pack(pady=(0, 0))
        
        ctk.CTkLabel(
            main_layout, text="Advanced Face Recognition Attendance Management System",
            font=ctk.CTkFont(size=14), text_color="gray50"
        ).pack(pady=(0, 10)) # Reduced padding
        
        # Feature cards
        cards_frame = ctk.CTkFrame(main_layout, fg_color="transparent")
        cards_frame.pack(pady=0) # Removed padding around cards container
        
        features = [
            ("capture", "Multi-Face Detection", "Recognize multiple students simultaneously"),
            ("records", "Excel Reports", "Export with charts and formatting"),
            ("home", "Modern Interface", "Dark and light theme support"),
            ("admin", "Secure Authentication", "SHA-256 hashed passwords"),
        ]
        
        for i, (icon_key, title, desc) in enumerate(features):
            card = ctk.CTkFrame(cards_frame, width=220, height=130, corner_radius=12)
            card.grid(row=0, column=i, padx=10, pady=5)
            card.pack_propagate(False)
            
            # Icon
            if self.icons and icon_key in self.icons:
                ctk.CTkLabel(card, text="", image=self.icons[icon_key]).pack(pady=(15, 5))
                
            ctk.CTkLabel(
                card, text=title, font=ctk.CTkFont(size=14, weight="bold")
            ).pack(pady=(0, 5))
            
            ctk.CTkLabel(
                card, text=desc, font=ctk.CTkFont(size=11),
                text_color="gray50", wraplength=180
            ).pack(padx=10)
        
        # Quick stats - Increased top padding for gap (30 -> 50)
        stats_frame = ctk.CTkFrame(main_layout, fg_color="transparent")
        stats_frame.pack(pady=(50, 20)) 
        
        student_count = 0
        if os.path.exists("StudentDetails/StudentDetails.csv"):
            try:
                df = pd.read_csv("StudentDetails/StudentDetails.csv")
                student_count = len(df)
            except Exception:
                pass
        
        record_count = 0
        if os.path.exists("Attendance"):
            record_count = len([f for f in os.listdir("Attendance") if f.endswith('.csv')])
        
        stats = [
            ("Registered Students", str(student_count)),
            ("Attendance Records", str(record_count)),
            ("Model Status", "Trained" if self.face_engine.is_model_loaded else "Not Trained"),
        ]
        
        for i, (label, value) in enumerate(stats):
            stat_card = ctk.CTkFrame(stats_frame, width=180, height=80, corner_radius=10)
            stat_card.grid(row=0, column=i, padx=15)
            stat_card.pack_propagate(False)
            
            ctk.CTkLabel(stat_card, text=value, font=ctk.CTkFont(size=24, weight="bold"),
                        text_color=self.ACCENT_COLOR).pack(pady=(15, 2))
            ctk.CTkLabel(stat_card, text=label, font=ctk.CTkFont(size=11),
                        text_color="gray50").pack()

    # ========================================================================
    # ANIMATION LOGIC
    # ========================================================================
    def _start_animation(self):
        self.animation_running = True
        self.animation_start_time = time.time()
        self._animate_loop()
    
    def _stop_animation(self):
        self.animation_running = False
    
    def _animate_loop(self):
        if not self.animation_running or not hasattr(self, 'home_logo_label'):
            return
            
        try:
            # Floating effect using Sine wave
            elapsed = time.time() - self.animation_start_time
            # Amplitude 0.05 (move 5% up/down), Frequency 2 (2 seconds per cycle)
            new_rely = 0.5 + 0.05 * np.sin(2 * np.pi * elapsed / 2) 
            
            self.home_logo_label.place(relx=0.5, rely=new_rely, anchor="center")
            
            # Schedule next frame (approx 30 FPS -> 33ms)
            self.after(33, self._animate_loop)
        except Exception:
            self._stop_animation()
    
    # ========================================================================
    # CAPTURE IMAGES VIEW
    # ========================================================================
    def _show_capture(self):
        self._hide_all_frames()
        self._stop_animation() # Stop animation when leaving home
        frame = self.frames["capture"]
        self._clear_frame(frame)
        frame.pack(fill="both", expand=True, padx=40, pady=40)
        
        ctk.CTkLabel(
            frame, text="Student Registration",
            font=ctk.CTkFont(size=28, weight="bold")
        ).pack(pady=(20, 30))
        
        # Form
        form = ctk.CTkFrame(frame, fg_color="transparent")
        form.pack(pady=20)
        
        ctk.CTkLabel(form, text="Enrollment Number:", font=ctk.CTkFont(size=13)).grid(
            row=0, column=0, padx=15, pady=12, sticky="e")
        self.enroll_entry = ctk.CTkEntry(form, width=320, height=42, font=ctk.CTkFont(size=13))
        self.enroll_entry.grid(row=0, column=1, padx=15, pady=12)
        
        ctk.CTkLabel(form, text="Student Name:", font=ctk.CTkFont(size=13)).grid(
            row=1, column=0, padx=15, pady=12, sticky="e")
        self.name_entry = ctk.CTkEntry(form, width=320, height=42, font=ctk.CTkFont(size=13))
        self.name_entry.grid(row=1, column=1, padx=15, pady=12)
        
        ctk.CTkButton(
            frame, text="Start Image Capture", width=250, height=48,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._capture_images
        ).pack(pady=30)
        
        self.capture_status = ctk.CTkLabel(frame, text="", font=ctk.CTkFont(size=13))
        self.capture_status.pack()
    
    def _capture_images(self):
        enrollment = self.enroll_entry.get().strip()
        name = self.name_entry.get().strip()
        
        if not enrollment or not name:
            messagebox.showerror("Validation Error", "Both Enrollment and Name are required.")
            return
        
        self.capture_status.configure(text="Initializing camera...", text_color="gray50")
        self.update()
        
        def capture_thread():
            try:
                cam_id = self.config_manager.config.get("camera_id", 0)
                cam = cv2.VideoCapture(cam_id)
                
                if not cam.isOpened():
                    self.after(0, lambda: self.capture_status.configure(
                        text="Camera access failed", text_color=self.ERROR_COLOR))
                    return
                
                cascade = self.face_engine.face_cascade
                os.makedirs("TrainingImage", exist_ok=True)
                
                sample_count = 0
                max_samples = self.config_manager.config.get("images_per_student", 70)
                
                while sample_count < max_samples:
                    ret, frame = cam.read()
                    if not ret:
                        break
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = cascade.detectMultiScale(gray, 1.3, 5)
                    
                    for (x, y, w, h) in faces:
                        sample_count += 1
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)
                        
                        filename = f"TrainingImage/{name}.{enrollment}.{sample_count}.jpg"
                        cv2.imwrite(filename, gray[y:y+h, x:x+w])
                        
                        cv2.putText(frame, f"Captured: {sample_count}/{max_samples}",
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
                    
                    cv2.imshow("Capture - Press Q to quit", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cam.release()
                cv2.destroyAllWindows()
                
                # Save student details
                os.makedirs("StudentDetails", exist_ok=True)
                csv_path = "StudentDetails/StudentDetails.csv"
                file_exists = os.path.exists(csv_path)
                
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['Enrollment', 'Name', 'Date', 'Time'])
                    writer.writerow([enrollment, name,
                                    datetime.now().strftime('%Y-%m-%d'),
                                    datetime.now().strftime('%H:%M:%S')])
                
                self.after(0, lambda: self.capture_status.configure(
                    text=f"Successfully captured {sample_count} images for {name}",
                    text_color=self.SUCCESS_COLOR))
                
            except Exception as e:
                self.after(0, lambda: self.capture_status.configure(
                    text=f"Error: {str(e)}", text_color=self.ERROR_COLOR))
        
        threading.Thread(target=capture_thread, daemon=True).start()
    
    def _on_train_model(self):
        """Handle model training."""
        progress_win = ctk.CTkToplevel(self)
        progress_win.title("Training Model")
        progress_win.geometry("400x150")
        progress_win.resizable(False, False)
        
        # Center the window
        progress_win.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 400) // 2
        y = self.winfo_y() + (self.winfo_height() - 150) // 2
        progress_win.geometry(f"+{x}+{y}")
        
        ctk.CTkLabel(progress_win, text="Training in progress...",
                    font=ctk.CTkFont(size=14)).pack(pady=20)
        
        progress = ctk.CTkProgressBar(progress_win, width=300, mode="indeterminate")
        progress.pack(pady=10)
        progress.start()
        
        status_label = ctk.CTkLabel(progress_win, text="", font=ctk.CTkFont(size=12))
        status_label.pack(pady=10)
        
        def train_thread():
            success, message = self.face_engine.train_model()
            progress.stop()
            
            if success:
                self.after(0, lambda: status_label.configure(text=message, text_color=self.SUCCESS_COLOR))
                self.after(2000, progress_win.destroy)
            else:
                self.after(0, lambda: status_label.configure(text=message, text_color=self.ERROR_COLOR))
                self.after(3000, progress_win.destroy)
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    # ========================================================================
    # ATTENDANCE VIEW
    # ========================================================================
    def _show_attendance(self):
        self._hide_all_frames()
        frame = self.frames["attendance"]
        self._clear_frame(frame)
        frame.pack(fill="both", expand=True, padx=40, pady=40)
        
        ctk.CTkLabel(
            frame, text="Take Attendance",
            font=ctk.CTkFont(size=28, weight="bold")
        ).pack(pady=(20, 30))
        
        # Subject input
        input_frame = ctk.CTkFrame(frame, fg_color="transparent")
        input_frame.pack(pady=15)
        
        ctk.CTkLabel(input_frame, text="Subject:", font=ctk.CTkFont(size=13)).pack(side="left", padx=10)
        self.subject_entry = ctk.CTkEntry(input_frame, width=280, height=40, font=ctk.CTkFont(size=13))
        self.subject_entry.pack(side="left", padx=10)
        
        # Buttons
        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(pady=25)
        
        ctk.CTkButton(
            btn_frame, text="Start Recognition", width=180, height=44,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._start_recognition
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            btn_frame, text="Export to Excel", width=180, height=44,
            font=ctk.CTkFont(size=13), fg_color=self.SUCCESS_COLOR,
            hover_color="#388E3C", command=self._export_current_session
        ).pack(side="left", padx=10)
        
        self.att_status = ctk.CTkLabel(frame, text="", font=ctk.CTkFont(size=13))
        self.att_status.pack(pady=15)
    
    def _start_recognition(self):
        subject = self.subject_entry.get().strip()
        if not subject:
            messagebox.showerror("Validation Error", "Please enter a subject name.")
            return
        
        self.current_subject = subject
        self.attendance_session = pd.DataFrame(columns=['Enrollment', 'Name', 'Date', 'Time'])
        
        if not self.face_engine.load_model():
            messagebox.showerror("Model Error", "Please train the model first.")
            return
        
        self.att_status.configure(text="Starting recognition...", text_color="gray50")
        self.update()
        
        def recognition_thread():
            try:
                cam = cv2.VideoCapture(self.config_manager.config.get("camera_id", 0))
                if not cam.isOpened():
                    self.after(0, lambda: self.att_status.configure(
                        text="Camera access failed", text_color=self.ERROR_COLOR))
                    return
                
                duration = self.config_manager.config.get("capture_duration", 20)
                end_time = time.time() + duration
                recognized_ids = set()
                
                while time.time() < end_time:
                    ret, frame = cam.read()
                    if not ret:
                        break
                    
                    detections = self.face_engine.detect_faces(frame)
                    frame = self.face_engine.draw_detections(frame, detections)
                    
                    for det in detections:
                        if det["recognized"] and det["id"] not in recognized_ids:
                            recognized_ids.add(det["id"])
                            now = datetime.now()
                            self.attendance_session.loc[len(self.attendance_session)] = [
                                det["id"], det["name"],
                                now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')
                            ]
                    
                    remaining = max(0, int(end_time - time.time()))
                    info = f"Time: {remaining}s | Detected: {len(recognized_ids)}"
                    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
                    
                    cv2.imshow("Attendance - Press Q to stop", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cam.release()
                cv2.destroyAllWindows()
                
                # Save CSV
                os.makedirs("Attendance", exist_ok=True)
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                csv_path = f"Attendance/{subject}_{timestamp}.csv"
                self.attendance_session.to_csv(csv_path, index=False)
                
                count = len(self.attendance_session)
                self.after(0, lambda: self.att_status.configure(
                    text=f"Recorded {count} students. Saved to: {csv_path}",
                    text_color=self.SUCCESS_COLOR))
                
            except Exception as e:
                self.after(0, lambda: self.att_status.configure(
                    text=f"Error: {str(e)}", text_color=self.ERROR_COLOR))
        
        threading.Thread(target=recognition_thread, daemon=True).start()
    
    def _export_current_session(self):
        if self.attendance_session.empty:
            messagebox.showwarning("No Data", "No attendance data to export.")
            return
        
        try:
            filepath = export_to_excel(self.attendance_session, self.current_subject or "Attendance")
            messagebox.showinfo("Export Successful", f"Exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))
    
    # ========================================================================
    # RECORDS VIEW
    # ========================================================================
    def _show_records(self):
        self._hide_all_frames()
        frame = self.frames["records"]
        self._clear_frame(frame)
        frame.pack(fill="both", expand=True, padx=40, pady=40)
        
        ctk.CTkLabel(
            frame, text="Attendance Records",
            font=ctk.CTkFont(size=28, weight="bold")
        ).pack(pady=(20, 30))
        
        if not os.path.exists("Attendance"):
            ctk.CTkLabel(frame, text="No records found.", text_color="gray50").pack(pady=50)
            return
        
        files = sorted([f for f in os.listdir("Attendance") if f.endswith('.csv')], reverse=True)
        
        if not files:
            ctk.CTkLabel(frame, text="No attendance records found.", text_color="gray50").pack(pady=50)
            return
        
        scroll = ctk.CTkScrollableFrame(frame, width=800, height=450)
        scroll.pack(pady=15)
        
        for filename in files[:30]:
            row = ctk.CTkFrame(scroll, height=50)
            row.pack(fill="x", pady=4, padx=10)
            row.pack_propagate(False)
            
            ctk.CTkLabel(row, text=filename, font=ctk.CTkFont(size=12)).pack(side="left", padx=15)
            
            ctk.CTkButton(
                row, text="Export Excel", width=110, height=32,
                font=ctk.CTkFont(size=11), fg_color=self.SUCCESS_COLOR,
                command=lambda f=filename: self._export_csv_file(f)
            ).pack(side="right", padx=10, pady=8)
    
    def _export_csv_file(self, filename: str):
        try:
            df = pd.read_csv(f"Attendance/{filename}")
            subject = filename.split('_')[0]
            filepath = export_to_excel(df, subject)
            messagebox.showinfo("Export Successful", f"Exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))
    
    # ========================================================================
    # ADMIN PANEL
    # ========================================================================
    def _show_admin_login(self):
        """Show admin login dialog."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Admin Login")
        dialog.geometry("380x280")
        dialog.resizable(False, False)
        
        # Center the dialog
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 380) // 2
        y = self.winfo_y() + (self.winfo_height() - 280) // 2
        dialog.geometry(f"+{x}+{y}")
        
        # Wait for window to be visible before grab_set
        dialog.after(100, lambda: self._setup_login_dialog(dialog))
    
    def _setup_login_dialog(self, dialog):
        """Setup login dialog content after window is visible."""
        try:
            dialog.grab_set()
        except Exception:
            pass
        
        ctk.CTkLabel(
            dialog, text="Administrator Login",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(30, 25))
        
        ctk.CTkLabel(dialog, text="Password:", font=ctk.CTkFont(size=12)).pack()
        password_entry = ctk.CTkEntry(dialog, show="*", width=260, height=40, font=ctk.CTkFont(size=13))
        password_entry.pack(pady=12)
        password_entry.focus()
        
        error_label = ctk.CTkLabel(dialog, text="", font=ctk.CTkFont(size=11), text_color=self.ERROR_COLOR)
        error_label.pack()
        
        def verify():
            if self.config_manager.verify_password(password_entry.get()):
                dialog.destroy()
                self._show_admin_panel()
            else:
                error_label.configure(text="Invalid password")
                password_entry.delete(0, 'end')
        
        ctk.CTkButton(
            dialog, text="Login", width=150, height=40,
            font=ctk.CTkFont(size=13, weight="bold"), command=verify
        ).pack(pady=15)
        
        password_entry.bind("<Return>", lambda e: verify())
    
    def _show_admin_panel(self):
        """Show admin panel after successful login."""
        panel = ctk.CTkToplevel(self)
        panel.title("Admin Panel")
        panel.geometry("650x550")
        
        # Center
        panel.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 650) // 2
        y = self.winfo_y() + (self.winfo_height() - 550) // 2
        panel.geometry(f"+{x}+{y}")
        
        ctk.CTkLabel(
            panel, text="Admin Panel",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(pady=(25, 20))
        
        # Registered students
        ctk.CTkLabel(panel, text="Registered Students:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=30)
        
        # Header Row
        header_row = ctk.CTkFrame(panel, fg_color="transparent")
        header_row.pack(fill="x", padx=35, pady=(5, 0))
        
        headers = [("ID", 60), ("Name", 150), ("Date", 100), ("Time", 100)]
        for i, (text, width) in enumerate(headers):
            lbl = ctk.CTkLabel(header_row, text=text, width=width, font=ctk.CTkFont(size=12, weight="bold"), anchor="w")
            lbl.pack(side="left", padx=5)
        
        student_frame = ctk.CTkScrollableFrame(panel, width=580, height=280)
        student_frame.pack(pady=5, padx=30)
        
        student_csv = "StudentDetails/StudentDetails.csv"
        if os.path.exists(student_csv):
            try:
                df = pd.read_csv(student_csv)
                for _, row in df.iterrows():
                    row_frame = ctk.CTkFrame(student_frame, height=35)
                    row_frame.pack(fill="x", pady=2)
                    
                    # Store values
                    enroll = str(row['Enrollment'])
                    name = str(row['Name'])
                    date = str(row['Date']) if 'Date' in row and pd.notna(row['Date']) else "-"
                    time = str(row['Time']) if 'Time' in row and pd.notna(row['Time']) else "-"
                    
                    # Columns
                    ctk.CTkLabel(row_frame, text=enroll, width=60, anchor="w").pack(side="left", padx=5)
                    ctk.CTkLabel(row_frame, text=name, width=150, anchor="w").pack(side="left", padx=5)
                    ctk.CTkLabel(row_frame, text=date, width=100, anchor="w").pack(side="left", padx=5)
                    ctk.CTkLabel(row_frame, text=time, width=100, anchor="w").pack(side="left", padx=5)
                    
            except Exception:
                ctk.CTkLabel(student_frame, text="Error loading student data", text_color="gray50").pack()
        else:
            ctk.CTkLabel(student_frame, text="No students registered", text_color="gray50").pack(pady=20)
        
        # Change password section
        ctk.CTkLabel(panel, text="Change Password:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=30, pady=(20, 5))
        
        pass_frame = ctk.CTkFrame(panel, fg_color="transparent")
        pass_frame.pack(pady=5)
        
        new_pass = ctk.CTkEntry(pass_frame, show="*", width=260, height=38, placeholder_text="Enter new password")
        new_pass.pack(side="left", padx=10)
        
        def change_password():
            pwd = new_pass.get().strip()
            if len(pwd) < 4:
                messagebox.showerror("Invalid Password", "Password must be at least 4 characters.")
                return
            self.config_manager.set_password(pwd)
            messagebox.showinfo("Success", "Password updated successfully.")
            new_pass.delete(0, 'end')
        
        ctk.CTkButton(
            pass_frame, text="Update", width=100, height=38,
            command=change_password
        ).pack(side="left", padx=10)
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    def _toggle_theme(self):
        new_theme = "dark" if self.theme_switch.get() else "light"
        ctk.set_appearance_mode(new_theme)
        self.config_manager.set_theme(new_theme)
    
    def _on_closing(self):
        self.destroy()


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    try:
        app = SmartAttendanceApp()
        app.mainloop()
    except KeyboardInterrupt:
        print("\nApplication closed by user.")
        try:
            if 'app' in locals():
                app.destroy()
        except:
            pass
    except Exception as e:
        print(f"An error occurred: {e}")

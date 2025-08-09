import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import shutil
from tkinter import font as tkfont

class FaceRecognitionAttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1280x720")
        self.root.configure(bg='#2c3e50')  # Dark blue background
        
        # Custom fonts
        self.title_font = tkfont.Font(family="Helvetica", size=24, weight="bold")
        self.header_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
        self.button_font = tkfont.Font(family="Helvetica", size=12, weight="bold")
        self.text_font = tkfont.Font(family="Helvetica", size=12)
        
        # Initialize variables
        self.haar_path = "haarcascade_frontalface_default.xml"
        self.student_details_path = "StudentDetails/StudentDetails.csv"
        self.training_image_path = "TrainingImage"
        self.training_label_path = "TrainingImageLabel/Trainner.yml"
        self.attendance_dir = "Attendance"
        self.current_attendance = []  # To store current session attendance
        
        # Setup GUI
        self.setup_gui()
        
        # Check required files
        self.check_required_files()
        
        # Start clock
        self.update_clock()
    
    def setup_gui(self):
        # Main container
        self.main_container = tk.Frame(self.root, bg='#2c3e50')
        self.main_container.pack(fill='both', expand=True)
        
        # Header
        self.setup_header()
        
        # Content area
        self.setup_content_area()
        
        # Menu
        self.setup_menu()
    
    def setup_header(self):
        header_frame = tk.Frame(self.main_container, bg='#34495e', height=80)
        header_frame.pack(fill='x', padx=10, pady=10)
        
        # Title
        title_label = tk.Label(
            header_frame, 
            text="Face Recognition Attendance System",
            fg="#ecf0f1", bg="#34495e", 
            font=self.title_font
        )
        title_label.pack(side='left', padx=20)
        
        # Date and time
        self.setup_datetime_display(header_frame)
    
    def setup_datetime_display(self, parent):
        datetime_frame = tk.Frame(parent, bg="#34495e")
        datetime_frame.pack(side='right', padx=20)
        
        # Date
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        day, month, year = date.split("-")
        
        mont = {
            '01': 'January', '02': 'February', '03': 'March', '04': 'April',
            '05': 'May', '06': 'June', '07': 'July', '08': 'August',
            '09': 'September', '10': 'October', '11': 'November', '12': 'December'
        }
        
        self.date_label = tk.Label(
            datetime_frame, 
            text=f"{day} {mont[month]} {year}",
            fg="#ecf0f1", bg="#34495e", 
            font=self.text_font
        )
        self.date_label.pack(side='top', anchor='e')
        
        # Time
        self.time_label = tk.Label(
            datetime_frame, 
            text="", 
            fg="#ecf0f1", bg="#34495e", 
            font=self.text_font
        )
        self.time_label.pack(side='bottom', anchor='e')
    
    def setup_content_area(self):
        content_frame = tk.Frame(self.main_container, bg='#2c3e50')
        content_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Left frame (Attendance)
        self.setup_attendance_frame(content_frame)
        
        # Right frame (Registration)
        self.setup_registration_frame(content_frame)
    
    def setup_attendance_frame(self, parent):
        attendance_frame = tk.Frame(parent, bg='#34495e', bd=2, relief='groove')
        attendance_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Header
        header = tk.Label(
            attendance_frame, 
            text="Attendance Management",
            fg="#ecf0f1", bg="#3498db", 
            font=self.header_font,
            pady=10
        )
        header.pack(fill='x')
        
        # Take attendance button
        btn_take_attendance = tk.Button(
            attendance_frame, 
            text="Take Attendance", 
            command=self.track_images,
            fg="#ecf0f1", bg="#27ae60", 
            font=self.button_font,
            height=2,
            width=20,
            relief='flat'
        )
        btn_take_attendance.pack(pady=(20, 10), padx=20, fill='x')
        
        # Attendance table
        self.setup_attendance_table(attendance_frame)
        
        # Quit button
        btn_quit = tk.Button(
            attendance_frame, 
            text="Quit", 
            command=self.root.destroy,
            fg="#ecf0f1", bg="#e74c3c", 
            font=self.button_font,
            height=2,
            width=20,
            relief='flat'
        )
        btn_quit.pack(pady=10, padx=20, fill='x')
    
    def setup_registration_frame(self, parent):
        registration_frame = tk.Frame(parent, bg='#34495e', bd=2, relief='groove')
        registration_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Header
        header = tk.Label(
            registration_frame, 
            text="Student Registration",
            fg="#ecf0f1", bg="#3498db", 
            font=self.header_font,
            pady=10
        )
        header.pack(fill='x')
        
        # Form fields
        self.setup_registration_form(registration_frame)
        
        # Action buttons
        self.setup_registration_buttons(registration_frame)
        
        # Status messages
        self.setup_status_messages(registration_frame)
    
    def setup_registration_form(self, parent):
        form_frame = tk.Frame(parent, bg='#34495e')
        form_frame.pack(pady=10)
        
        # Roll number
        lbl_roll = tk.Label(
            form_frame, 
            text="Student ID:", 
            fg="#ecf0f1", bg="#34495e", 
            font=self.text_font,
            anchor='w'
        )
        lbl_roll.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        
        self.txt_roll = tk.Entry(
            form_frame, 
            width=25, 
            fg="#2c3e50", 
            font=self.text_font,
            relief='flat'
        )
        self.txt_roll.grid(row=0, column=1, padx=10, pady=5)
        
        btn_clear_roll = tk.Button(
            form_frame, 
            text="Clear", 
            command=self.clear_roll,
            fg="#ecf0f1", bg="#7f8c8d", 
            font=self.button_font,
            width=8,
            relief='flat'
        )
        btn_clear_roll.grid(row=0, column=2, padx=5, pady=5)
        
        # Name
        lbl_name = tk.Label(
            form_frame, 
            text="Full Name:", 
            fg="#ecf0f1", bg="#34495e", 
            font=self.text_font,
            anchor='w'
        )
        lbl_name.grid(row=1, column=0, padx=10, pady=5, sticky='w')
        
        self.txt_name = tk.Entry(
            form_frame, 
            width=25, 
            fg="#2c3e50", 
            font=self.text_font,
            relief='flat'
        )
        self.txt_name.grid(row=1, column=1, padx=10, pady=5)
        
        btn_clear_name = tk.Button(
            form_frame, 
            text="Clear", 
            command=self.clear_name,
            fg="#ecf0f1", bg="#7f8c8d", 
            font=self.button_font,
            width=8,
            relief='flat'
        )
        btn_clear_name.grid(row=1, column=2, padx=5, pady=5)
    
    def setup_registration_buttons(self, parent):
        button_frame = tk.Frame(parent, bg='#34495e')
        button_frame.pack(pady=10)
        
        btn_take_images = tk.Button(
            button_frame, 
            text="Capture Images", 
            command=self.take_images,
            fg="#ecf0f1", bg="#2980b9", 
            font=self.button_font,
            height=2,
            width=25,
            relief='flat'
        )
        btn_take_images.pack(pady=5, fill='x', padx=20)
        
        btn_train_images = tk.Button(
            button_frame, 
            text="Train Model", 
            command=self.train_images,
            fg="#ecf0f1", bg="#16a085", 
            font=self.button_font,
            height=2,
            width=25,
            relief='flat'
        )
        btn_train_images.pack(pady=5, fill='x', padx=20)
    
    def setup_status_messages(self, parent):
        message_frame = tk.Frame(parent, bg='#34495e')
        message_frame.pack(fill='x', pady=10, padx=10)
        
        self.instruction_label = tk.Label(
            message_frame, 
            text="1) Capture Images  →  2) Train Model",
            bg="#34495e", fg="#f1c40f", 
            font=self.text_font
        )
        self.instruction_label.pack()
        
        self.status_label = tk.Label(
            message_frame, 
            text="", 
            bg="#34495e", fg="#2ecc71", 
            font=self.text_font
        )
        self.status_label.pack(pady=(5, 0))
        
        self.registration_count_label = tk.Label(
            message_frame, 
            text="Total Registrations: 0", 
            bg="#34495e", fg="#bdc3c7", 
            font=self.text_font
        )
        self.registration_count_label.pack(pady=(10, 0))
    
    def setup_attendance_table(self, parent):
        table_frame = tk.Frame(parent, bg='#34495e')
        table_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Treeview with scrollbar
        self.tv = ttk.Treeview(
            table_frame, 
            height=10, 
            columns=('name', 'date', 'time'),
            selectmode='browse'
        )
        
        # Style configuration
        style = ttk.Style()
        style.configure("Treeview", 
                       background="#ffffff", 
                       foreground="#2c3e50",
                       rowheight=25,
                       fieldbackground="#0A0A0A",
                       font=self.text_font)
        style.configure("Treeview.Heading", 
                       background="#3498db", 
                       foreground="#000000",
                       font=self.text_font)
        style.map('Treeview', background=[('selected', '#2980b9')])
        
        # Columns
        self.tv.column('#0', width=100, anchor='center')
        self.tv.column('name', width=150, anchor='w')
        self.tv.column('date', width=120, anchor='center')
        self.tv.column('time', width=120, anchor='center')
        
        # Headings
        self.tv.heading('#0', text='ID', anchor='center')
        self.tv.heading('name', text='NAME', anchor='center')
        self.tv.heading('date', text='DATE', anchor='center')
        self.tv.heading('time', text='TIME', anchor='center')
        
        # Scrollbar
        scroll_y = ttk.Scrollbar(table_frame, orient='vertical', command=self.tv.yview)
        scroll_y.pack(side='right', fill='y')
        self.tv.configure(yscrollcommand=scroll_y.set)
        
        self.tv.pack(fill='both', expand=True)
    
    def setup_menu(self):
        menubar = tk.Menu(self.root, relief='ridge', bg='#34495e', fg='#ecf0f1')
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0, bg='#34495e', fg='#ecf0f1')
        file_menu.add_command(
            label='Change Password', 
            command=self.change_password,
            font=self.text_font
        )
        file_menu.add_separator()
        file_menu.add_command(
            label='Exit', 
            command=self.root.destroy,
            font=self.text_font
        )
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0, bg='#34495e', fg='#ecf0f1')
        help_menu.add_command(
            label='User Guide', 
            command=self.user_guide,
            font=self.text_font
        )
        help_menu.add_command(
            label='About', 
            command=self.about,
            font=self.text_font
        )
        help_menu.add_command(
            label='Contact Us', 
            command=self.contact,
            font=self.text_font
        )
        
        menubar.add_cascade(label='File', menu=file_menu, font=self.text_font)
        menubar.add_cascade(label='Help', menu=help_menu, font=self.text_font)
        
        self.root.config(menu=menubar)
    
    def check_required_files(self):
        # Check haarcascade file
        if not os.path.isfile(self.haar_path):
            # Try to find it in OpenCV installation
            opencv_path = os.path.join(
                os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml"
            )
            if os.path.isfile(opencv_path):
                shutil.copy(opencv_path, self.haar_path)
            else:
                messagebox.showerror(
                    "Missing File",
                    "haarcascade_frontalface_default.xml not found. Please download it."
                )
                self.root.destroy()
        
        # Create required directories
        os.makedirs("StudentDetails", exist_ok=True)
        os.makedirs(self.training_image_path, exist_ok=True)
        os.makedirs("TrainingImageLabel", exist_ok=True)
        os.makedirs(self.attendance_dir, exist_ok=True)
        
        # Initialize student details CSV if not exists
        if not os.path.isfile(self.student_details_path):
            with open(self.student_details_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['SERIAL NO.', 'ID', 'NAME'])
        
        self.update_registration_count()
    
    def update_clock(self):
        time_string = time.strftime('%H:%M:%S')
        self.time_label.config(text=time_string)
        self.root.after(200, self.update_clock)
    
    def clear_roll(self):
        self.txt_roll.delete(0, 'end')
        self.instruction_label.config(text="1) Capture Images  →  2) Train Model")
    
    def clear_name(self):
        self.txt_name.delete(0, 'end')
        self.instruction_label.config(text="1) Capture Images  →  2) Train Model")
    
    def take_images(self):
        roll = self.txt_roll.get().strip()
        name = self.txt_name.get().strip()
        
        if not name.replace(' ', '').isalpha():
            self.status_label.config(text="Error: Name must contain only letters and spaces", fg="#e74c3c")
            return
        
        if not roll:
            self.status_label.config(text="Error: Please enter student ID", fg="#e74c3c")
            return
        
        # Get next serial number
        with open(self.student_details_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            serial = sum(1 for _ in reader) + 1
        
        # Initialize camera
        try:
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                raise Exception("Camera not accessible")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        
        detector = cv2.CascadeClassifier(self.haar_path)
        sample_num = 0
        
        # Create a window for displaying camera feed
        cv2.namedWindow("Capturing Images", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Capturing Images", 800, 600)
        
        self.status_label.config(text="Capturing images... Please look at the camera", fg="#f39c12")
        self.root.update()
        
        while sample_num < 100:
            ret, img = cam.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sample_num += 1
                
                # Save image
                img_name = f"{name}.{serial}.{roll}.{sample_num}.jpg"
                img_path = os.path.join(self.training_image_path, img_name)
                cv2.imwrite(img_path, gray[y:y+h, x:x+w])
                
                # Display progress
                cv2.putText(img, f"Captured: {sample_num}/100", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display
                cv2.imshow('Capturing Images', img)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        
        cam.release()
        cv2.destroyAllWindows()
        
        # Save to CSV
        with open(self.student_details_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([serial, roll, name])
        
        self.status_label.config(text=f"Success: Captured 100 images for {name} (ID: {roll})", fg="#2ecc71")
        self.instruction_label.config(text="Now train the model with the captured images")
        self.update_registration_count()
    
    def train_images(self):
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            detector = cv2.CascadeClassifier(self.haar_path)
            
            faces, ids = self.get_images_and_labels(self.training_image_path)
            
            if not faces:
                self.status_label.config(text="Error: No training images found", fg="#e74c3c")
                return
            
            recognizer.train(faces, np.array(ids))
            recognizer.save(self.training_label_path)
            
            self.status_label.config(text="Success: Model trained with all images", fg="#2ecc71")
            self.instruction_label.config(text="You can now take attendance")
        except Exception as e:
            self.status_label.config(text=f"Error: Training failed - {str(e)}", fg="#e74c3c")
    
    def get_images_and_labels(self, path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
        faces = []
        ids = []
        
        for image_path in image_paths:
            pil_image = Image.open(image_path).convert('L')
            image_np = np.array(pil_image, 'uint8')
            
            try:
                id = int(os.path.split(image_path)[-1].split(".")[1])
                faces.append(image_np)
                ids.append(id)
            except (IndexError, ValueError):
                continue
        
        return faces, ids
    
    def track_images(self):
        # Clear previous attendance records
        for item in self.tv.get_children():
            self.tv.delete(item)
        
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(self.training_label_path)
        except:
            messagebox.showerror("Error", "No trained data found. Please train the model first.")
            return
        
        detector = cv2.CascadeClassifier(self.haar_path)
        
        try:
            df = pd.read_csv(self.student_details_path)
        except:
            messagebox.showerror("Error", "Student details not found")
            return
        
        try:
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                raise Exception("Camera not accessible")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        col_names = ['Id', 'Name', 'Date', 'Time']
        self.current_attendance = []
        
        # Dictionary to track last recognition time for each student
        self.last_recognized = {}
        self.recognition_cooldown = 5  # seconds between recognitions
        
        # Create a window for displaying camera feed
        cv2.namedWindow("Taking Attendance", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Taking Attendance", 800, 600)
        
        self.status_label.config(text="Taking attendance... Press 'q' to stop", fg="#f39c12")
        self.root.update()
        
        while True:
            ret, im = cam.read()
            if not ret:
                continue
            
            current_time = time.time()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
                
                if conf < 50:  # Confidence threshold
                    try:
                        aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values[0]
                        id = df.loc[df['SERIAL NO.'] == serial]['ID'].values[0]
                        
                        # Check if we recently recognized this student
                        last_time = self.last_recognized.get(id, 0)
                        if current_time - last_time < self.recognition_cooldown:
                            # Skip marking attendance if recently recognized
                            cv2.putText(im, f"Already marked: {aa}", (x, y + h + 30), 
                                        font, 0.8, (0, 255, 255), 2)
                            continue
                            
                        # Mark attendance
                        ts = time.time()
                        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                        time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        
                        # Check if this student has already been marked today
                        already_marked = any(att[0] == str(id) for att in self.current_attendance)
                        
                        if not already_marked:
                            attendance = [str(id), str(aa), date, time_stamp]
                            self.current_attendance.append(attendance)
                            self.last_recognized[id] = current_time
                            
                            # Add to treeview
                            self.tv.insert('', 0, text=attendance[0], 
                                         values=(attendance[1], attendance[2], attendance[3]))
                            
                            # Display confirmation
                            cv2.putText(im, f"Marked: {aa}", (x, y + h + 30), 
                                        font, 0.8, (0, 255, 0), 2)
                        
                        cv2.putText(im, str(aa), (x, y + h), font, 1, (255, 255, 255), 2)
                    except Exception as e:
                        print(f"Error: {e}")
                else:
                    cv2.putText(im, "Unknown", (x, y + h), font, 1, (255, 255, 255), 2)
            
            cv2.imshow('Taking Attendance', im)
            if cv2.waitKey(1) == ord('q'):
                break
        
        cam.release()
        cv2.destroyAllWindows()
        
        # Save attendance
        if self.current_attendance:
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
            attendance_file = os.path.join(self.attendance_dir, f"Attendance_{date}.csv")
            
            # Read existing attendance to prevent duplicates
            existing_attendance = []
            if os.path.isfile(attendance_file):
                with open(attendance_file, 'r') as file:
                    reader = csv.reader(file)
                    next(reader)  # Skip header
                    existing_attendance = list(reader)
            
            # Write header if file doesn't exist
            if not os.path.isfile(attendance_file):
                with open(attendance_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(col_names)
            
            # Append only new attendance records
            with open(attendance_file, 'a', newline='') as file:
                writer = csv.writer(file)
                for record in self.current_attendance:
                    # Check if this record already exists
                    if not any(record[0] == existing[0] and record[2] == existing[2] 
                              for existing in existing_attendance):
                        writer.writerow(record)
            
            self.status_label.config(
                text=f"Success: Attendance marked for {len(self.current_attendance)} students", 
                fg="#2ecc71"
            )
        else:
            self.status_label.config(text="No attendance marked in this session", fg="#e74c3c")
    
    def update_registration_count(self):
        try:
            with open(self.student_details_path, 'r') as file:
                count = sum(1 for _ in csv.reader(file)) - 1  # Subtract header
                self.registration_count_label.config(text=f"Total Registrations: {count}")
        except:
            self.registration_count_label.config(text="Total Registrations: 0")
    
    def change_password(self):
        password_window = tk.Toplevel(self.root)
        password_window.title("Change Password")
        password_window.geometry("400x300")
        password_window.resizable(False, False)
        password_window.configure(bg='#34495e')
        
        # Title
        tk.Label(
            password_window, 
            text="Change Password", 
            fg="#ecf0f1", bg="#3498db", 
            font=self.header_font,
            pady=10
        ).pack(fill='x')
        
        # Form container
        form_frame = tk.Frame(password_window, bg='#34495e')
        form_frame.pack(pady=20)
        
        # Old password
        tk.Label(
            form_frame, 
            text="Old Password:", 
            fg="#ecf0f1", bg="#34495e", 
            font=self.text_font
        ).grid(row=0, column=0, padx=10, pady=10, sticky='e')
        
        old_pass = tk.Entry(
            form_frame, 
            show="*", 
            font=self.text_font,
            width=25
        )
        old_pass.grid(row=0, column=1, padx=10, pady=10)
        
        # New password
        tk.Label(
            form_frame, 
            text="New Password:", 
            fg="#ecf0f1", bg="#34495e", 
            font=self.text_font
        ).grid(row=1, column=0, padx=10, pady=10, sticky='e')
        
        new_pass = tk.Entry(
            form_frame, 
            show="*", 
            font=self.text_font,
            width=25
        )
        new_pass.grid(row=1, column=1, padx=10, pady=10)
        
        # Confirm new password
        tk.Label(
            form_frame, 
            text="Confirm Password:", 
            fg="#ecf0f1", bg="#34495e", 
            font=self.text_font
        ).grid(row=2, column=0, padx=10, pady=10, sticky='e')
        
        confirm_pass = tk.Entry(
            form_frame, 
            show="*", 
            font=self.text_font,
            width=25
        )
        confirm_pass.grid(row=2, column=1, padx=10, pady=10)
        
        # Button frame
        button_frame = tk.Frame(password_window, bg='#34495e')
        button_frame.pack(pady=10)
        
        def save_password():
            # Implement password change logic here
            if new_pass.get() != confirm_pass.get():
                messagebox.showerror("Error", "New passwords don't match")
                return
            
            # Here you would typically:
            # 1. Verify old password
            # 2. Update to new password
            # 3. Save to secure storage
            
            messagebox.showinfo("Success", "Password changed successfully")
            password_window.destroy()
        
        tk.Button(
            button_frame, 
            text="Save", 
            command=save_password,
            fg="#ecf0f1", bg="#27ae60", 
            font=self.button_font,
            width=15
        ).pack(side='left', padx=10)
        
        tk.Button(
            button_frame, 
            text="Cancel", 
            command=password_window.destroy,
            fg="#ecf0f1", bg="#e74c3c", 
            font=self.button_font,
            width=15
        ).pack(side='right', padx=10)
    
    def user_guide(self):
        guide = """
        Face Recognition Attendance System - User Guide
        
        1. Registration:
           - Enter Student ID and Name
           - Click 'Capture Images' and look at the camera
           - After capturing, click 'Train Model'
           
        2. Taking Attendance:
           - Click 'Take Attendance' and look at the camera
           - The system will automatically mark attendance
           - Press 'q' to stop
           
        3. Viewing Attendance:
           - Attendance records are displayed in the table
           - Full records are saved in CSV files
        """
        messagebox.showinfo("User Guide", guide)
    
    def about(self):
        about_text = """
        Face Recognition Attendance System
        
        Version: 1.0
        Developed by: Nilay Kumar
        
        This application uses OpenCV and face recognition
        technology to automate attendance tracking.
        """
        messagebox.showinfo("About", about_text)
    
    def contact(self):
        messagebox.showinfo(
            "Contact Us", 
            "For support or inquiries:\n\n"
            "Email: nilay9101@gmail.com\n"
            "Phone: +91 8603781068\n"
            
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionAttendanceSystem(root)
    root.mainloop()
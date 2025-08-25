import sys, os, json, traceback
from datetime import datetime
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QFileDialog, QLabel, QSpacerItem, QSizePolicy,
    QComboBox, QLineEdit, QTableWidget, QTableWidgetItem, QTabWidget,
    QHeaderView
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ai_service import analyze_billboard  


STYLESHEET = """
    /* Main window */
    QWidget {
        font-family: 'Segoe UI', Arial, sans-serif;
        background-color: #f5f7fa;
    }
    
    /* Input fields */
    QLineEdit, QComboBox {
        border: 1px solid #d1d9e6;
        border-radius: 6px;
        padding: 7px;
        font-size: 14px;
        background: white;
    }
    
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left-width: 1px;
        border-left-color: #d1d9e6;
        border-left-style: solid;
    }
    
    /* Buttons */
    QPushButton {
        background-color: #4a6fa5;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        min-width: 100px;
    }
    
    QPushButton:hover {
        background-color: #3a5a85;
    }
    
    QPushButton:disabled {
        background-color: #b3c0d4;
        color: #666;
    }
    
    /* Labels */
    QLabel {
        font-size: 14px;
    }
    
    /* Cards */
    .card {
        background: white;
        border: 1px solid #e0e6ed;
        border-radius: 8px;
        padding: 12px;
    }
    
    /* Image containers */
    .image-container {
        background: white;
        border: 1px solid #e0e6ed;
        border-radius: 8px;
    }
    
    /* Status badges */
    .status-badge {
        font-weight: bold;
        padding: 6px 12px;
        border-radius: 12px;
        font-size: 13px;
        qproperty-alignment: 'AlignCenter';
    }
    
    /* Table */
    QTableWidget {
        border: 1px solid #e0e6ed;
        border-radius: 8px;
        background: white;
    }
    
    QHeaderView::section {
        background-color: #4a6fa5;
        color: white;
        padding: 8px;
        font-weight: bold;
    }
    
    QTableWidget::item {
        padding: 5px;
    }
"""


class AnalyzeWorker(QThread):
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, pil_image, city, area_type):
        super().__init__()
        self.pil_image = pil_image
        self.city = city
        self.area_type = area_type
    
    def run(self):
        try:
            result = analyze_billboard(self.pil_image, self.city, self.area_type, visualize=True)
            if not isinstance(result, dict):
                raise RuntimeError("Invalid result format from analysis")
            result["city"] = self.city
            result["area_type"] = self.area_type
            self.finished_signal.emit(result)
        except Exception as e:
            tb = traceback.format_exc()
            self.error_signal.emit(f"Analysis Error:\n{str(e)}\n\n{tb}")


class BillboardAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Billboard Compliance Analyzer")
        self.resize(1300, 840)
        self.setStyleSheet(STYLESHEET)
        
        
        self.latest_result = None
        self.worker = None
        self.history = []
        
       
        self.initUI()
    
    def initUI(self):
       
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        self.setLayout(main_layout)
        
       
        self.tabs = QTabWidget()
        self.createAnalyzerTab()
        self.createDashboardTab()
        
        main_layout.addWidget(self.tabs)
    
    def createAnalyzerTab(self):
        self.analyzer_tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        self.analyzer_tab.setLayout(layout)
        
       
        title = QLabel("Billboard Compliance Analysis")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title)
        
       
        input_card = QWidget()
        input_card.setProperty("class", "card")
        input_layout = QHBoxLayout()
        input_layout.setSpacing(15)
        input_card.setLayout(input_layout)
        
        
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Your email")
        self.email_input.setFixedWidth(250)
        
        self.location_input = QLineEdit()
        self.location_input.setPlaceholderText("Location details")
        self.location_input.setFixedWidth(300)
        
        self.city_dropdown = QComboBox()
        self.city_dropdown.addItems(["Chennai","Delhi","Mumbai","Pune","Kolkata","Bengaluru","Hyderabad","Ahmedabad"])
        self.city_dropdown.setFixedWidth(180)
        
        self.area_dropdown = QComboBox()
        self.area_dropdown.addItems(["Urban","Rural","Highway"])
        self.area_dropdown.setFixedWidth(140)
        
        self.pick_button = QPushButton("Select Image")
        self.pick_button.clicked.connect(self.on_pick_click)
        
        
        input_layout.addWidget(QLabel("Email:"))
        input_layout.addWidget(self.email_input)
        input_layout.addWidget(QLabel("Location:"))
        input_layout.addWidget(self.location_input)
        input_layout.addWidget(QLabel("City:"))
        input_layout.addWidget(self.city_dropdown)
        input_layout.addWidget(QLabel("Area:"))
        input_layout.addWidget(self.area_dropdown)
        input_layout.addWidget(self.pick_button)
        
        layout.addWidget(input_card)
        
       
        analysis_area = QHBoxLayout()
        analysis_area.setSpacing(20)
        
       
        image_col = QVBoxLayout()
        image_col.setSpacing(15)
        
        self.orig_card = QLabel("Original image will appear here")
        self.orig_card.setProperty("class", "image-container")
        self.orig_card.setAlignment(Qt.AlignCenter)
        self.orig_card.setMinimumSize(640, 380)
        
        self.crop_card = QLabel("Cropped billboard preview")
        self.crop_card.setProperty("class", "image-container")
        self.crop_card.setAlignment(Qt.AlignCenter)
        self.crop_card.setMinimumSize(640, 240)
        
        image_col.addWidget(self.orig_card)
        image_col.addWidget(self.crop_card)
        
        analysis_area.addLayout(image_col)
        
        
        results_card = QWidget()
        results_card.setProperty("class", "card")
        results_layout = QVBoxLayout()
        results_layout.setSpacing(10)
        results_card.setLayout(results_layout)
        
        self.status_badge = QLabel("Pending analysis")
        self.status_badge.setProperty("class", "status-badge")
        self.status_badge.setStyleSheet("background: #bdc3c7;")
        
        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        self.result_area.setMinimumHeight(300)
        
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.save_cropped_btn = QPushButton("Save Cropped")
        self.save_cropped_btn.clicked.connect(self.save_cropped)
        self.save_cropped_btn.setEnabled(False)
        
        self.export_json_btn = QPushButton("Export Report")
        self.export_json_btn.clicked.connect(self.export_json)
        self.export_json_btn.setEnabled(False)
        
        self.loader_label = QLabel()
        
        button_layout.addWidget(self.save_cropped_btn)
        button_layout.addWidget(self.export_json_btn)
        button_layout.addWidget(self.loader_label)
        
        results_layout.addWidget(self.status_badge)
        results_layout.addWidget(self.result_area)
        results_layout.addLayout(button_layout)
        
        analysis_area.addWidget(results_card)
        layout.addLayout(analysis_area)
        
        self.tabs.addTab(self.analyzer_tab, "Analyzer")
    
    def createDashboardTab(self):
        self.dashboard_tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        self.dashboard_tab.setLayout(layout)
        
        title = QLabel("Reports Dashboard")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title)
        
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels([
            "Date", "City", "Area", "Status", "Reason", 
            "Width (m)", "Height (m)", "Reporter", "Location"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)
        
        self.tabs.addTab(self.dashboard_tab, "Dashboard")
    
    def on_pick_click(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Billboard Image", 
            "", 
            "Images (*.png *.jpg *.jpeg)"
        )
        if path:
            self.analyze_image(path)
    
    def analyze_image(self, path):
        try:
            
            pixmap = QPixmap(path).scaled(
                self.orig_card.width(), 
                self.orig_card.height(), 
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.orig_card.setPixmap(pixmap)
            self.crop_card.setText("Analyzing...")
            
            
            try:
                pil_img = Image.open(path).convert("RGB")
            except Exception as e:
                raise Exception(f"Failed to open image: {str(e)}")
            
            
            self.set_ui_busy(True)
            self.worker = AnalyzeWorker(
                pil_img, 
                self.city_dropdown.currentText(),
                self.area_dropdown.currentText()
            )
            self.worker.finished_signal.connect(self.on_analysis_complete)
            self.worker.error_signal.connect(self.on_analysis_error)
            self.worker.start()
            
        except Exception as e:
            self.show_error(str(e))
            self.set_ui_busy(False)
    
    def on_analysis_complete(self, result):
        self.latest_result = result
        self.set_ui_busy(False)
        
        
        self.latest_result["user_email"] = self.email_input.text().strip()
        self.latest_result["location"] = self.location_input.text().strip()
        
        
        cropped_path = result.get("croppedImageUrl")
        if cropped_path and os.path.exists(cropped_path):
            cropped_pix = QPixmap(cropped_path).scaled(
                self.crop_card.width(),
                self.crop_card.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.crop_card.setPixmap(cropped_pix)
            self.save_cropped_btn.setEnabled(True)
        else:
            self.crop_card.setText("No cropped image available")
        
       
        analysis = result.get("analysis", {})
        legal_status = self.normalize_legal_status(analysis.get("legal_status"))
        reason = analysis.get("reason", "No specific reason provided")
        width = float(analysis.get("billboard_width_m", 0.0))
        height = float(analysis.get("billboard_height_m", 0.0))
        area_pct = float(result.get("billboard_area_percentage", 0.0))
        oversized = bool(result.get("oversized", False))
        angle = float(analysis.get("billboard_angle_deg", 0.0))
        
        
        if legal_status == "illegal":
            self.status_badge.setText("VIOLATION")
            self.status_badge.setStyleSheet("background: #e74c3c; color: white;")
        else:
            self.status_badge.setText("COMPLIANT")
            self.status_badge.setStyleSheet("background: #2ecc71; color: white;")
        
        
        results_html = f"""
        <div style="text-align: center;">
            <h3><b>Analysis Results</b></h3>
            <p><b>Status:</b> <span style="color: {'#e74c3c' if legal_status == 'illegal' else '#2ecc71'};">
                <b>{'NON-COMPLIANT' if legal_status == 'illegal' else 'COMPLIANT'}</b>
            </span></p>
            <p><b>Reason:</b> <b>{reason}</b></p>
            <p><b>Dimensions:</b> <b>{width:.2f}m × {height:.2f}m</b></p>
            <p><b>Angle:</b> <b>{angle:.2f}°</b></p>
            <p><b>Area Coverage:</b> <b>{area_pct:.1f}%</b></p>
            <p><b>Oversized:</b> <b>{'Yes' if oversized else 'No'}</b></p>
        </div>
        """
        self.result_area.setHtml(results_html)
        self.export_json_btn.setEnabled(True)
        
        
        self.add_to_history(result, legal_status, reason, width, height)
    
    def normalize_legal_status(self, status):
        if not status:
            return "unknown"
        status = str(status).lower()
        if status in ("illegal", "prohibited", "not compliant"):
            return "illegal"
        if status in ("legal", "compliant", "allowed"):
            return "legal"
        return status
    
    def add_to_history(self, result, status, reason, width, height):
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        items = [
            QTableWidgetItem(datetime.now().strftime("%Y-%m-%d %H:%M")),
            QTableWidgetItem(result.get("city", "")),
            QTableWidgetItem(result.get("area_type", "")),
            QTableWidgetItem("VIOLATION" if status == "illegal" else "COMPLIANT"),
            QTableWidgetItem(reason),
            QTableWidgetItem(f"{width:.2f}"),
            QTableWidgetItem(f"{height:.2f}"),
            QTableWidgetItem(self.latest_result.get("user_email", "")),
            QTableWidgetItem(self.latest_result.get("location", ""))
        ]
        
        for col, item in enumerate(items):
            self.table.setItem(row, col, item)
    
    def on_analysis_error(self, error_msg):
        self.show_error(error_msg)
        self.set_ui_busy(False)
    
    def set_ui_busy(self, is_busy):
        self.pick_button.setEnabled(not is_busy)
        self.loader_label.setText("Analyzing..." if is_busy else "")
    
    def show_error(self, message):
        self.result_area.setPlainText(message)
        self.status_badge.setText("ERROR")
        self.status_badge.setStyleSheet("background: #e74c3c; color: white;")
    
    def save_cropped(self):
        if not self.latest_result:
            return
            
        cropped_path = self.latest_result.get("croppedImageUrl")
        if not cropped_path or not os.path.exists(cropped_path):
            self.show_error("No cropped image available to save")
            return
            
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Cropped Image",
            f"cropped_billboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            "Images (*.png *.jpg *.jpeg)"
        )
        
        if save_path:
            try:
                Image.open(cropped_path).save(save_path)
                self.show_error(f"Cropped image saved to:\n{save_path}")
            except Exception as e:
                self.show_error(f"Failed to save image:\n{str(e)}")
    
    def export_json(self):
        if not self.latest_result:
            return
            
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Analysis Report",
            f"billboard_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )
        
        if save_path:
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(self.latest_result, f, indent=2, ensure_ascii=False)
                
                # Display the disclaimer after successful export
                self.result_area.append("\n\nNOTE: This report is generated for analytical purposes only. All data processing is performed with strict privacy protections.")
            except Exception as e:
                self.show_error(f"Failed to export report:\n{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = BillboardAnalyzer()
    window.show()
    
    sys.exit(app.exec_())

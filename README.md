# Billboard Compliance Analyzer  

Our project tackles the challenge of **unauthorized and non-compliant billboards** in cities.  
We built an AI-powered system that automatically detects billboards and checks if they comply with **government rules on size, placement, and content**.  

The solution is designed with a strong focus on **mobile accessibility**, so that anyone can report a billboard instantly using their phone.  



## Key Features
- **Mobile App(with credit and reporting system)** for quick on-the-go reporting.  
- **Desktop App (PyQt5 GUI,)** for detailed analysis and offline use,with credit and reporting system. 
- **Command-line Tool** for fast testing by developers.  
- Automated detection using a **custom-trained YOLOv8 model**.  
- OCR-based text extraction with filters for prohibited, abusive, or misleading content.  
- Visual content moderation (detects NSFW or inappropriate images).  
- Generates clear compliance reports and credits citizens who help flag illegal billboards.  



## Project Files
- `ai_service.py` – Core AI logic (detection, OCR, compliance checks).  
- `streamlit_app.py` – Mobile app for reporting and dashboard view.  
- `gui_app.py` – Desktop app for detailed compliance analysis.  
- `test.py` – CLI tool for quick checks.  
- `requirements.txt` – Dependencies list.  
- *(Optional)* `yolo12m.pt` – Custom YOLOv8 model file (if offline).  

---

## Installation
1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

2. Run the mobile app (recommended for hackathon demo):  
   ```bash
   streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
   ```  

   ### How to open on mobile:
   - Find your laptop’s **IP address**:  
     - Windows: `ipconfig` → look for IPv4 (e.g. `192.168.1.23`)  
     - Linux/Mac: `ifconfig` → find `inet` under Wi-Fi adapter  
   - Connect your phone to the **same Wi-Fi / hotspot** as your laptop  
   - Open browser on phone and visit:  
     ```
     http://<your-laptop-ip>:8501
     ```  
     Example: `http://192.168.1.23:8501`  

3. For other modes:  
   - Desktop app:  
     ```bash
     python gui_app.py
     ```  
   - Command-line:  
     ```bash
     python test.py <image_path> <state> <area_type> [visualize]
     ```  

Example:  
```bash
python test.py sample.jpg "Tamil Nadu" Urban true
```  

---


- The **mobile app is the primary interface**, making it simple for citizens and judges to test directly from their phones.  
- Internet is required for model download (from HuggingFace). If offline, include `yolo12m.pt` in the folder and adjust `ai_service.py`.  
- To make testing easier, include a couple of sample images in the submission.  

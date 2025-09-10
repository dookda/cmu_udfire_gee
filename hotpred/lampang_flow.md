```mermaid
graph TD
    subgraph "1. การนำเข้าข้อมูล (Data Ingestion)"
        direction LR
        ExternalDB["🗄️<br/><b>ฐานข้อมูลภายนอก</b><br/>(JHCIS, 43 แฟ้ม)"] --> DIL["⚙️<br/><b>Data Integration Layer</b>"]
        DIL -- "<i>แปลงและจัดเก็บ</i>" --> CoreDB[("💾<br/><b>Core Database</b>")]
    end

    subgraph "2. การโต้ตอบของผู้ใช้งาน (User Interaction)"
        direction TB
        User["👩‍⚕️<br/><b>เจ้าหน้าที่สาธารณสุข</b>"] -- "<i>ล็อกอิน/บันทึกข้อมูลวัคซีน</i>" --> Frontend["💻<br/><b>Web Application</b>"]
        Frontend -- "<i>1. ส่งคำขอ</i>" --> APIGW["🚪<br/><b>API Gateway</b>"]
        APIGW -- "<i>2. ตรวจสอบสิทธิ์ & ส่งต่อ</i>" --> VMS["💉<br/><b>Vaccine Management Service</b>"]
        VMS -- "<i>3. ประมวลผล & บันทึก</i>" --> CoreDB
        VMS -- "<i>4. สร้าง Event แจ้งเตือน</i>" --> MB[("📨<br/><b>Message Broker</b>")]
    end

    subgraph "3. การทำงานของ AI และการแจ้งเตือน (AI & Notification)"
        direction TB
        subgraph "AI Analytics"
            CoreDB -- "<i>1. ดึงข้อมูลเป็นรอบ</i>" --> AIS["🤖<br/><b>Analytics & AI Service</b>"]
            AIS -- "<i>2. ส่งผลวิเคราะห์กลับ</i>" --> Frontend
        end

        subgraph "Notification"
             MB -- "<i>1. เมื่อถึงเวลานัดหมาย</i>" --> NS["💬<br/><b>Notification Service</b>"]
             NS -- "<i>2. ส่ง SMS/ข้อความ</i>" --> Patient["👨‍👩‍👧‍👦<br/><b>ผู้ป่วย/ผู้ปกครอง</b>"]
        end
    end

    subgraph "4. การแสดงผล (Dashboard & Reporting)"
        direction TB
        Admin["👨‍💼<br/><b>ผู้บริหาร/เจ้าหน้าที่</b>"] -- "<i>1. เปิดดู Dashboard</i>" --> Frontend
        Frontend -- "<i>2. ขอข้อมูล</i>" --> APIGW
        APIGW -- "<i>3. ส่งต่อไปยัง</i>" --> RS["📈<br/><b>Reporting Service</b>"]
        RS -- "<i>4. ดึงข้อมูลจาก Services</i>" --> OtherServices["Vaccine, Obesity,<br/>Spatial Services"]
        RS -- "<i>5. สังเคราะห์ข้อมูล</i>" --> APIGW
        APIGW -- "<i>6. ส่งข้อมูลไปแสดงผล</i>" --> Frontend
    end

```
```mermaid
graph TD
    subgraph "1. การนำเข้าข้อมูล (Data Ingestion)"
        ExternalDB[ฐานข้อมูลภายนอก<br/>(JHCIS, 43 แฟ้ม)] --> DIL[Data Integration Layer]
        DIL -- แปลงและจัดเก็บ --> CoreDB[(Core Database)]
    end

    subgraph "2. การโต้ตอบของผู้ใช้งาน (User Interaction)"
        User[👩‍⚕️ เจ้าหน้าที่] -- ล็อกอิน/บันทึกข้อมูลวัคซีน --> Frontend[🌐 Web Application]
        Frontend -- 1. ส่งคำขอ --> APIGW[API Gateway]
        APIGW -- 2. ตรวจสอบสิทธิ์และส่งต่อ --> VMS[💉 Vaccine Management Service]
        VMS -- 3. ประมวลผลและบันทึก --> CoreDB
        VMS -- 4. สร้าง Event แจ้งเตือน --> MB[(Message Broker)]
    end

    subgraph "3. การทำงานของ AI และการแจ้งเตือน (AI & Notification)"
        style AIS fill:#d4edda
        style NS fill:#f8d7da
        
        subgraph "AI Analytics"
            AIS[🤖 Analytics & AI Service] -- 1. ดึงข้อมูลเป็นรอบ --> CoreDB
            AIS -- 2. ส่งผลวิเคราะห์ --> Frontend
        end

        subgraph "Notification"
            MB -- 1. เมื่อถึงเวลานัดหมาย --> NS[💬 Notification Service]
            NS -- 2. ส่ง SMS/ข้อความ --> Patient[👨‍👩‍👧‍👦 ผู้ป่วย/ผู้ปกครอง]
        end
    end

    subgraph "4. การแสดงผล (Dashboard & Reporting)"
        Admin[👨‍💼 ผู้บริหาร/เจ้าหน้าที่] -- 1. เปิดดู Dashboard --> Frontend
        Frontend -- 2. ขอข้อมูล --> APIGW
        APIGW -- 3. ส่งต่อไปยัง --> RS[📈 Reporting Service]
        RS -- 4. ดึงข้อมูลจาก Service ต่างๆ --> Services[Vaccine, Obesity,<br/>Spatial Services]
        RS -- 5. สังเคราะห์ข้อมูลและส่งกลับ --> APIGW
        APIGW -- 6. ส่งข้อมูลไปแสดงผล --> Frontend
    end
```
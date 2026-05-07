# 🧪 JMeter Load Testing — Local + Cloud

คู่มือใช้ Apache JMeter รัน load test ของ `/predict` ทั้งบน
**local uvicorn** และ **production HF Spaces**

---

## 0) ติดตั้ง JMeter (ครั้งเดียว)

ดาวน์โหลด **Apache JMeter 5.6.3+** จาก <https://jmeter.apache.org/download_jmeter.cgi>

แตก zip แล้วเพิ่ม `bin/` เข้า PATH (Windows):

```powershell
# ตัวอย่าง: ถ้าแตกไว้ที่ C:\Tools\apache-jmeter-5.6.3
$env:Path += ";C:\Tools\apache-jmeter-5.6.3\bin"

# Permanent (ครั้งเดียว) — รันใน PowerShell แบบ Administrator:
[Environment]::SetEnvironmentVariable(
    "Path",
    [Environment]::GetEnvironmentVariable("Path", "User") + ";C:\Tools\apache-jmeter-5.6.3\bin",
    "User"
)
```

ตรวจ:

```powershell
jmeter -v
# ควรขึ้น: Version 5.6.3 ...
```

> JMeter ต้องใช้ **Java 17+** ติดตั้งล่วงหน้า (`java -version` ตรวจดู)

---

## 1) เตรียม test image

```powershell
cd "C:\Users\User\Documents\Claude\Projects\Image Classification\sports-vit-mlops"
python scripts/make_test_image.py
```

ได้ไฟล์ `tests/fixtures/tennis.jpg` (~5 KB) ใช้แทนรูปจริงสำหรับยิง /predict ซ้ำ ๆ
ถ้าอยากใช้รูปจริง ก็เอามาวางทับชื่อเดิม

---

## 2) รัน LOCAL test

### 2.1 เปิด API ก่อน

ใน terminal A:

```powershell
cd "C:\Users\User\Documents\Claude\Projects\Image Classification\sports-vit-mlops"
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### 2.2 รัน JMeter ใน terminal B

```powershell
cd "C:\Users\User\Documents\Claude\Projects\Image Classification\sports-vit-mlops"
.\scripts\run_jmeter_local.ps1
```

ปรับพารามิเตอร์ได้:

```powershell
.\scripts\run_jmeter_local.ps1 -Threads 50 -Duration 120 -RampUp 20
```

ผลลัพธ์อยู่ที่:
```
jmeter/results/local_<timestamp>/
├── result.jtl              ← raw samples
├── summary.csv             ← summary report
├── aggregate.csv           ← aggregate stats
└── report/
    └── index.html          ← HTML dashboard ← เปิดอันนี้!
```

เปิด dashboard:
```powershell
Start-Process "jmeter/results/local_<timestamp>/report/index.html"
```

---

## 3) รัน CLOUD test (HF Spaces)

ก่อนรัน — ตรวจให้แน่ใจว่า Space ตื่น (เปิด browser ไปที่ `https://<user>-<space>.hf.space/health` ครั้งหนึ่งก่อน):

```powershell
.\scripts\run_jmeter_cloud.ps1 -HFUser tonnot3t3 -HFSpace sports-vit-api
```

ปรับพารามิเตอร์:

```powershell
.\scripts\run_jmeter_cloud.ps1 `
    -HFUser tonnot3t3 `
    -HFSpace sports-vit-api `
    -Threads 30 `
    -Duration 120 `
    -RampUp 30
```

> ⚠️ **HF Spaces (free tier) มี 2 vCPU + 16 GB RAM** — อย่ายิงหนักเกิน 30-50 threads ไม่งั้นจะ rate-limit หรือ timeout
>
> Space cold start ~ 30 วินาที (ครั้งแรกหลังหลับ); ครั้งต่อไปจะเร็ว

ผลลัพธ์อยู่ที่ `jmeter/results/cloud_<timestamp>/` — โครงเหมือน local

---

## 4) เปรียบเทียบ Local vs Cloud

ตัวอย่างตารางที่ควรเอาไปใส่ Project Report:

| Metric              | Local (CPU 2 workers) | Cloud (HF Space free) |
| ------------------- | --------------------- | --------------------- |
| Threads             | 30                    | 20                    |
| Duration            | 60 s                  | 60 s                  |
| Throughput (req/s)  | 18-22                 | 6-10                  |
| Median latency      | ~ 95 ms               | ~ 250 ms              |
| 95th percentile     | ~ 150 ms              | ~ 600 ms              |
| Error %             | 0.0%                  | 0.0–0.5%              |

> ตัวเลขจริงจะอยู่ในไฟล์ `summary.csv` / `aggregate.csv` หลังรันเสร็จ — capture screenshot ของ HTML dashboard ใส่ slide

---

## 5) Screenshot ที่ควรเก็บใส่รายงาน / สไลด์

จาก `report/index.html` ของแต่ละ run:

1. **APDEX score** (หน้าแรกของ dashboard)
2. **Statistics** (ตารางหลัก: median / 90% / 95% / 99% / throughput)
3. **Response Times Over Time** (graph เส้น)
4. **Active Threads Over Time**
5. **Response Times Distribution**

---

## 6) Troubleshooting

| อาการ                                     | แก้                                                       |
| ----------------------------------------- | ---------------------------------------------------------- |
| `jmeter : not recognized`                 | ยังไม่ได้เพิ่ม `bin/` เข้า PATH                            |
| `cannot reach http://localhost:7860`      | ลืมเปิด uvicorn ก่อน                                       |
| HTTP 503 ตอน warm-up                      | รอ 5–10 วิ ให้ ProcessPoolExecutor เริ่มเสร็จ              |
| HTTP 415 unsupported media                | MIME ของไฟล์ไม่ใช่ image/*; ตรวจ `tests/fixtures/tennis.jpg` |
| Cloud test timeout 30s                    | Space กำลัง cold-start; เปิด /health ใน browser ก่อนแล้วลองใหม่ |
| รายการ assertion JSONPath fail            | response shape เปลี่ยน; ตรวจ `$.predictions[0].label`       |

---

## 7) คำสั่งดิบ (ถ้าไม่อยากใช้ .ps1 wrapper)

### Local

```powershell
jmeter -n -t jmeter/load_test.jmx `
       -l jmeter/results/local.jtl `
       -e -o jmeter/results/local_report `
       "-Jhost=localhost" "-Jport=7860" "-Jscheme=http" `
       "-Jthreads=30" "-Jrampup=15" "-Jduration=60" `
       "-Jimage_path=tests/fixtures/tennis.jpg"
```

### Cloud

```powershell
jmeter -n -t jmeter/load_test.jmx `
       -l jmeter/results/cloud.jtl `
       -e -o jmeter/results/cloud_report `
       "-Jhost=tonnot3t3-sports-vit-api.hf.space" `
       "-Jport=443" "-Jscheme=https" `
       "-Jthreads=20" "-Jrampup=15" "-Jduration=60" `
       "-Jimage_path=tests/fixtures/tennis.jpg"
```

# 🚀 Deploy Guide — GitHub + Hugging Face Spaces

คู่มือแบบ step-by-step สำหรับเอา repository ขึ้น GitHub
และให้ CI/CD auto-deploy ลง Hugging Face Spaces

---

## 0) สิ่งที่ต้องเตรียม

- บัญชี [GitHub](https://github.com/)
- บัญชี [Hugging Face](https://huggingface.co/) (ฟรี)
- Git LFS ติดตั้งบนเครื่องแล้ว (`git lfs install` จะรันแค่ครั้งเดียว)
- ไฟล์ `onnx_models/vit_sports_int8.onnx` มีแล้ว (จาก `python scripts/optimize.py`)

---

## 1) สร้าง Hugging Face Space (Docker SDK)

1. ไปที่ <https://huggingface.co/new-space>
2. ตั้งค่า:
   - **Owner:** username ของคุณ (เช่น `outplaytkfunny`)
   - **Space name:** `sports-vit-api` *(หรือชื่ออื่นที่ชอบ — จดไว้)*
   - **License:** Apache-2.0
   - **Space SDK:** **Docker** → **Blank**
   - **Visibility:** Public (ฟรี) หรือ Private ก็ได้
3. กด **Create Space**

> URL ของ Space จะเป็น `https://huggingface.co/spaces/<USERNAME>/<SPACE_NAME>`
> และ endpoint สำหรับเรียก API จะเป็น `https://<USERNAME>-<SPACE_NAME>.hf.space`

---

## 2) สร้าง Hugging Face Access Token (write)

1. ไปที่ <https://huggingface.co/settings/tokens>
2. กด **New token**
3. ตั้งค่า:
   - **Name:** `github-actions-deploy` (อะไรก็ได้)
   - **Type:** **Write**
4. กด **Generate** → คัดลอก token เก็บไว้ (จะเห็นแค่ครั้งเดียว!)

---

## 3) สร้าง GitHub repository และ push

```powershell
cd "C:\Users\User\Documents\Claude\Projects\Image Classification\sports-vit-mlops"

# init git ถ้ายังไม่ได้ทำ
git init
git lfs install
git add .
git commit -m "Initial commit: Sports ViT MLOps project"

# สร้าง repo บน github (ผ่าน CLI หรือเว็บ) แล้วเชื่อม
# ตัวอย่างผ่านเว็บ: ไปที่ https://github.com/new → ตั้งชื่อ sports-vit-mlops (ห้าม init README)

git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USER>/sports-vit-mlops.git
git push -u origin main
```

> ✅ ไฟล์ `*.onnx` จะถูก track ผ่าน Git LFS โดยอัตโนมัติ (ดู `.gitattributes`)

---

## 4) ใส่ Secrets ให้ GitHub Actions

ไปที่ repo บน GitHub → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

เพิ่ม **3 ตัว** ต่อไปนี้:

| Name              | Value                              | ตัวอย่าง            |
| ----------------- | ---------------------------------- | ------------------- |
| `HF_TOKEN`        | HF write token จากขั้นตอน 2        | `hf_xxxxxxxxxxxx`   |
| `HF_USERNAME`     | username บน Hugging Face           | `outplaytkfunny`    |
| `HF_SPACE_NAME`   | ชื่อ Space ที่สร้างในขั้นตอน 1     | `sports-vit-api`    |

---

## 5) Trigger CI/CD

แค่ push commit ใดก็ได้ขึ้น `main`:

```powershell
git commit --allow-empty -m "ci: trigger deploy"
git push
```

ดูสถานะที่แท็บ **Actions** ของ repo:

1. **Lint & Unit Tests** — รัน `pytest`
2. **Build Docker image** — ตรวจ Dockerfile
3. **Deploy to Hugging Face Spaces** — push code ไปที่ Space (เฉพาะ branch `main` และเฉพาะตอน test ผ่าน)

ถ้า deploy job เสร็จ → กลับไปที่ Space, แท็บ **Logs** จะเห็น Docker build ของ HF กำลังรัน
รอประมาณ 3–8 นาที (ครั้งแรกนานสุดเพราะต้อง pull image)

---

## 6) ทดสอบ Production endpoint

เปิด browser ไปที่:

```
https://<HF_USERNAME>-<HF_SPACE_NAME>.hf.space/
```

จะเห็น **Web UI** สำหรับ upload หลายภาพในครั้งเดียว ✨

หรือเรียกผ่าน cURL:

```bash
curl -X POST \
  -F "file=@tennis.jpg" \
  https://<HF_USERNAME>-<HF_SPACE_NAME>.hf.space/predict
```

---

## 🔁 Workflow ปกติหลังจากนี้

```powershell
# แก้ code
git add .
git commit -m "feat: ..."
git push        # ← CI รัน → ถ้า test ผ่าน → auto-deploy ขึ้น HF
```

---

## 🛠 Troubleshooting

| อาการ                              | แก้                                                                 |
| ---------------------------------- | -------------------------------------------------------------------- |
| `git push` แล้ว `*.onnx` ใหญ่เกิน  | รัน `git lfs install` ก่อน commit; ตรวจ `git lfs ls-files`           |
| HF Space build ขึ้น "no Dockerfile"| ตรวจให้แน่ใจว่า `Dockerfile` อยู่ root และ `sdk: docker` ใน README   |
| HF Space ขึ้นแต่ /predict 503       | Space ยังโหลด ONNX อยู่ — รอสัก 30–60 วิ                            |
| GitHub Action `deploy` failed 403  | HF token ไม่ใช่ **write**; สร้างใหม่และอัพเดต `HF_TOKEN`             |
| Image > 5 MB ตอนทดสอบ              | ลดขนาด หรือเพิ่ม `MAX_IMAGE_BYTES` ใน env var                        |

---

## 🔧 ปรับ env vars บน HF Space (ถ้าต้องการ)

ไปที่ Space → **Settings** → **Variables and secrets** → เพิ่มเช่น:

- `WORKER_PROCESSES=2`
- `TOP_K=5`
- `MAX_IMAGE_BYTES=10485760`  *(10 MB)*
- `LOG_LEVEL=INFO`

Space จะ rebuild อัตโนมัติเมื่อแก้ค่า

---

🎉 เสร็จแล้ว — ทุก push ไปที่ `main` จะ deploy ขึ้น production โดยอัตโนมัติ

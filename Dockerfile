# 1. استخدام صورة بايثون خفيفة كأساس
FROM python:3.9-slim

# 2. إعداد مجلد العمل داخل الحاوية
WORKDIR /app

# 3. نسخ ملف المتطلبات وتثبيت المكتبات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. نسخ باقي ملفات المشروع (الكود والمودل)
COPY . .

# 5. فتح المنفذ 8000
EXPOSE 8000

# 6. أمر تشغيل الـ API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
import os
import requests

TARGET_DIR = r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\Transdiagnostic"
DATASET = "ds005237"

os.makedirs(TARGET_DIR, exist_ok=True)

# Read subjects from participants.tsv
with open(os.path.join(TARGET_DIR, "participants.tsv"), "r") as f:
    lines = f.read().strip().split("\n")

headers = lines[0].split("\t")
pid_idx = 0
for i, h in enumerate(headers):
    if "participant" in h.lower():
        pid_idx = i
        break

subjects = []
for line in lines[1:]:
    parts = line.split("\t")
    if len(parts) > pid_idx:
        subjects.append(parts[pid_idx].strip())

print(f"Found {len(subjects)} subjects")

# Try downloading BOLD files directly using OpenNeuro S3 URLs
# Format: https://s3.amazonaws.com/openneuro.org/ds005237/sub-XXX/ses-YYY/func/sub-XXX_ses-YYY_task-rest_bold.nii.gz

downloaded = 0
skipped = 0
failed = 0

for i, sub in enumerate(subjects):
    print(f"\n[{i+1}/{len(subjects)}] {sub}")
    
    # Try different session patterns
    found_any = False
    for ses in ["", "ses-01", "ses-1", "ses-2", "ses-baseline"]:
        for run in ["", "run-01", "run-1", "run-02"]:
            ses_path = f"{ses}/" if ses else ""
            run_part = f"_{run}" if run else ""
            ses_id = f"_{ses}" if ses else ""
            
            # .nii.gz file
            bold_name = f"{sub}{ses_id}_task-rest{run_part}_bold.nii.gz"
            json_name = f"{sub}{ses_id}_task-rest{run_part}_bold.json"
            
            for fname in [bold_name, json_name]:
                remote = f"{sub}/{ses_path}func/{fname}"
                url = f"https://s3.amazonaws.com/openneuro.org/{DATASET}/{remote}"
                
                local_dir = os.path.join(TARGET_DIR, sub, ses_path.replace("/", ""), "func") if ses else os.path.join(TARGET_DIR, sub, "func")
                os.makedirs(local_dir, exist_ok=True)
                local_path = os.path.join(local_dir, fname)
                
                if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                    skipped += 1
                    found_any = True
                    continue
                
                try:
                    rr = requests.head(url, timeout=10)
                    if rr.status_code == 200:
                        size_mb = int(rr.headers.get("Content-Length", 0)) / (1024*1024)
                        print(f"  Downloading: {remote} ({size_mb:.0f} MB)")
                        rr = requests.get(url, stream=True, timeout=600)
                        if rr.status_code == 200:
                            with open(local_path, "wb") as out:
                                for chunk in rr.iter_content(8192):
                                    out.write(chunk)
                            downloaded += 1
                            found_any = True
                except:
                    pass
    
    if not found_any and i < 3:
        print(f"  No files found - trying OpenNeuro direct URL...")
        # Try OpenNeuro direct URL
        url = f"https://openneuro.org/crn/datasets/{DATASET}/files/{sub}/func"
        try:
            rr = requests.get(url, timeout=10)
            print(f"  Direct URL status: {rr.status_code}")
        except:
            pass

print(f"\n=== DONE ===")
print(f"Downloaded: {downloaded}")
print(f"Skipped (already exist): {skipped}")
print(f"Failed: {failed}")
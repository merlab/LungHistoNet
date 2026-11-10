import requests
import json
import os
from pathlib import Path
from collections import defaultdict
import csv
import argparse


"""
All project
python dgdc-WSI-downloader.py --download-type tissue --projects all: Downloads tissue slides and metadata for all 11 projects.
python gdc-WSI-downloader.py --download-type diagnostic --projects all: Downloads diagnostic slides and metadata for all 11 projects.
python gdc-WSI-downloader.py --download-type both --projects all: Downloads both tissue and diagnostic slides and metadata for all 11 projects.
python gdc-WSI-downloader.py --download-type none --projects all: Downloads metadata only for all 11 projects.

Single project:
python gdc-WSI-downloader.py --download-type tissue --projects TCGA-BRCA: Downloads tissue slides and metadata for TCGA-BRCA only.
Multiple projects:
python gdc-WSI-downloader.py --download-type both --projects TCGA-BRCA,TCGA-LUAD,TCGA-OV: Downloads both tissue and diagnostic slides and metadata for TCGA-BRCA, TCGA-LUAD, and TCGA-OV.
"""
GDC_API_ENDPOINT = "https://api.gdc.cancer.gov"
BASE_DIR = "tcga_data"
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
SLIDES_DIR = os.path.join(BASE_DIR, "slides")
ALL_PROJECTS = [
    "TCGA-BRCA", "TCGA-LUAD", "TCGA-LUSC", "TCGA-OV", "TCGA-GBM",
    "TCGA-UCEC", "TCGA-COAD", "TCGA-PRAD", "TCGA-THCA", "TCGA-HNSC", "TCGA-LIHC"
]

def create_directories(project_id, download_type):
    project_metadata_dir = os.path.join(METADATA_DIR, project_id)
    Path(project_metadata_dir).mkdir(parents=True, exist_ok=True)
    project_slides_dir = None
    if download_type != "none":
        project_slides_dir = os.path.join(SLIDES_DIR, project_id)
        Path(project_slides_dir).mkdir(parents=True, exist_ok=True)
    return project_metadata_dir, project_slides_dir

def get_manifest(project_id):
    url = f"{GDC_API_ENDPOINT}/files"
    params = {
        "filters": json.dumps({
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "cases.project.project_id", "value": project_id}},
                {"op": "=", "content": {"field": "data_category", "value": "Biospecimen"}},
                {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}}
            ]
        }),
        "fields": "file_id,file_name,md5sum,case_id,file_size,data_format,experimental_strategy,cases.submitter_id",
        "format": "json",
        "size": 10000
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def group_by_patient(files, download_type):
    patient_slides = defaultdict(list)
    for file in files:
        experimental_strategy = file.get("experimental_strategy", "")
        if download_type == "tissue" and experimental_strategy != "Tissue Slide":
            continue
        if download_type == "diagnostic" and experimental_strategy != "Diagnostic Slide":
            continue
        case_id = file.get("case_id")
        submitter_id = file.get("cases", [{}])[0].get("submitter_id", "Unknown")
        identifier = case_id or submitter_id
        patient_slides[identifier].append(file)
    return patient_slides

def save_metadata(project_id, identifier, slides, project_metadata_dir):
    output_path = os.path.join(project_metadata_dir, f"{identifier}.json")
    with open(output_path, "w") as f:
        json.dump(slides, f, indent=2)
    print(f"Saved metadata for {project_id}, patient {identifier} ({len(slides)} slides)")

def download_file(project_id, file_id, file_name, identifier, md5sum, project_slides_dir):
    patient_dir = os.path.join(project_slides_dir, identifier)
    Path(patient_dir).mkdir(exist_ok=True)
    output_path = os.path.join(patient_dir, file_name)
    if os.path.exists(output_path):
        print(f"Skipping {file_name} for {project_id}, patient {identifier}, already exists")
        return
    url = f"{GDC_API_ENDPOINT}/data/{file_id}"
    response = requests.get(url)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded {file_name} for {project_id}, patient {identifier}")

def generate_project_summary_csv(project_id, patient_slides):
    csv_path = os.path.join(BASE_DIR, f"{project_id}_summary.csv")
    total_files = sum(len(slides) for slides in patient_slides.values())
    total_size_bytes = sum(file.get("file_size", 0) for slides in patient_slides.values() for file in slides)
    total_size_mb = total_size_bytes / (1024 * 1024)
    formats = set(file.get("data_format", "Unknown") for slides in patient_slides.values() for file in slides)
    tissue_slides = sum(1 for slides in patient_slides.values() for file in slides if file.get("experimental_strategy") == "Tissue Slide")
    diagnostic_slides = sum(1 for slides in patient_slides.values() for file in slides if file.get("experimental_strategy") == "Diagnostic Slide")
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Project", "Total Patients", "Total Slides", "Tissue Slides", "Diagnostic Slides", "Total Size (MB)", "File Formats"])
        writer.writerow([project_id, len(patient_slides), total_files, tissue_slides, diagnostic_slides, f"{total_size_mb:.2f}", ", ".join(formats)])
        writer.writerow([])
        writer.writerow(["Patient ID", "Number of Slides", "Tissue Slides", "Diagnostic Slides", "Size (MB)"])
        for identifier, slides in sorted(patient_slides.items()):
            patient_size_mb = sum(file.get("file_size", 0) for file in slides) / (1024 * 1024)
            patient_tissue = sum(1 for file in slides if file.get("experimental_strategy") == "Tissue Slide")
            patient_diagnostic = sum(1 for file in slides if file.get("experimental_strategy") == "Diagnostic Slide")
            writer.writerow([identifier, len(slides), patient_tissue, patient_diagnostic, f"{patient_size_mb:.2f}"])
    
    print(f"\nGenerated project summary CSV for {project_id}: {csv_path}")
    print(f"Summary for {project_id}:")
    print(f"Total number of patients: {len(patient_slides)}")
    print(f"Total number of slide images: {total_files}")
    print(f"Tissue slides: {tissue_slides}")
    print(f"Diagnostic slides: {diagnostic_slides}")
    print(f"Total size of slides: {total_size_mb:.2f} MB")
    print(f"File formats: {', '.join(formats)}")
    print("Note: Slide dimensions (width, height) are not available in GDC metadata. Download SVS files and use OpenSlide to extract dimensions.")
    
    return {
        "project": project_id,
        "total_patients": len(patient_slides),
        "total_slides": total_files,
        "tissue_slides": tissue_slides,
        "diagnostic_slides": diagnostic_slides,
        "total_size_mb": total_size_mb,
        "file_formats": ", ".join(formats)
    }

def generate_all_projects_summary_csv(project_summaries):
    csv_path = os.path.join(BASE_DIR, "all_tcga_projects_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Project", "Total Patients", "Total Slides", "Tissue Slides", "Diagnostic Slides", "Total Size (MB)", "File Formats"])
        for summary in project_summaries:
            writer.writerow([
                summary["project"],
                summary["total_patients"],
                summary["total_slides"],
                summary["tissue_slides"],
                summary["diagnostic_slides"],
                f"{summary['total_size_mb']:.2f}",
                summary["file_formats"]
            ])
    print(f"\nGenerated all projects summary CSV: {csv_path}")

def download_tcga_slides(download_type="both", projects="all"):
    if download_type not in ["tissue", "diagnostic", "both", "none"]:
        raise ValueError("download_type must be 'tissue', 'diagnostic', 'both', or 'none'")
    
    if projects.lower() == "all":
        project_list = ALL_PROJECTS
    else:
        project_list = [p.strip() for p in projects.split(",")]
        invalid_projects = [p for p in project_list if p not in ALL_PROJECTS]
        if invalid_projects:
            raise ValueError(f"Invalid project(s): {', '.join(invalid_projects)}. Available projects: {', '.join(ALL_PROJECTS)}")
    
    project_summaries = []
    for project_id in project_list:
        print(f"\nProcessing {project_id}...")
        project_metadata_dir, project_slides_dir = create_directories(project_id, download_type)
        manifest = get_manifest(project_id)
        files = manifest["data"]["hits"]
        patient_slides = group_by_patient(files, download_type if download_type != "none" else "both")
        
        for identifier, slides in patient_slides.items():
            save_metadata(project_id, identifier, slides, project_metadata_dir)
            if download_type != "none":
                for file in slides:
                    if (download_type == "tissue" and file.get("experimental_strategy") != "Tissue Slide") or \
                       (download_type == "diagnostic" and file.get("experimental_strategy") != "Diagnostic Slide"):
                        continue
                    file_id = file["file_id"]
                    file_name = file["file_name"]
                    md5sum = file["md5sum"]
                    download_file(project_id, file_id, file_name, identifier, md5sum, project_slides_dir)
        
        project_summary = generate_project_summary_csv(project_id, patient_slides)
        project_summaries.append(project_summary)
    
    generate_all_projects_summary_csv(project_summaries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TCGA metadata and slides by type and projects.")
    parser.add_argument("--download-type", choices=["tissue", "diagnostic", "both", "none"], default="both",
                        help="Type of slides to download: 'tissue', 'diagnostic', 'both', or 'none' for metadata only")
    parser.add_argument("--projects", default="all",
                        help="Comma-separated TCGA project IDs (e.g., TCGA-BRCA,TCGA-LUAD) or 'all' for all available projects")
    args = parser.parse_args()
    
    download_tcga_slides(args.download_type, args.projects)
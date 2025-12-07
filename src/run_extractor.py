
from preprocessing.pdf_extractor import add_json_to_csv, extract_pdf_to_json


if __name__ == "__main__":


    pdf_file = "C:\\Users\\yesmine\\Desktop\\Tanit\\data\\raw\\sample.pdf"
    csv_file = "C:\\Users\\yesmine\\Desktop\\Tanit\\data\\processed\\new_patients.csv"    
    print("🔵 EXTRACTION PDF → JSON (COPIE EXACTE)")
    print("="*80)
    
    # Extraction
    json_path = extract_pdf_to_json(pdf_file, "extraction_pdf.json")
    
    if json_path:
        print(f"\n✅ EXTRACTION pdf to json TERMINÉE !")
        print(f"📁 Fichier JSON : {json_path}")
    else:
        print("\n❌ ÉCHEC DE L'EXTRACTION")
    
    json_file = "extraction_pdf.json"  
    
    print("\n🔵 EXTRACTION ET AJOUT DE DONNÉES JSON VERS CSV")
    print("="*80)
    
    success = add_json_to_csv(
        json_path=json_file,
        csv_path="C:\\Users\\yesmine\\Desktop\\Tanit\\data\\raw\\patients.csv"
    )
    
    if success:
        print("\n✅ OPÉRATION  ajout au csv TERMINÉE AVEC SUCCÈS !")
    else:
        print("\n❌ ÉCHEC DE L'OPÉRATION")

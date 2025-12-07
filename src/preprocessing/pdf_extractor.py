import json
from PyPDF2 import PdfReader
import google.generativeai as genai
import csv
from pathlib import Path
from datetime import datetime


# Configuration Gemini
genai.configure(api_key="AIzaSyDI5WuqCClZoHjL8Clap0RnocQB7mnPUT4")
model = genai.GenerativeModel('gemini-2.0-flash')


def extract_pdf_to_json(pdf_path, output_json="extraction_pdf.json"):
    """
    Extrait TOUTES les informations du PDF et fait une copie EXACTE des tableaux en JSON
    """
    print("\n" + "="*80)
    print("📝 EXTRACTION PDF → JSON (COPIE EXACTE)")
    print("="*80 + "\n")
    
    try:
        print(f"📂 Lecture du PDF : {pdf_path}")

        # Extraire le texte du PDF
        pdf_reader = PdfReader(pdf_path)
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"

        print(f"✓ PDF lu : {len(pdf_reader.pages)} page(s)")
        print(f"✓ Longueur texte : {len(full_text)} caractères\n")

        # Prompt pour copie EXACTE
        prompt = f"""
Tu es un expert en extraction de données depuis des PDF médicaux.

Texte extrait du PDF :
```
{full_text}
```

🎯 **MISSION : COPIE EXACTE DES DONNÉES**
=========================================

⚠️⚠️⚠️ RÈGLE ABSOLUE : COPIE EXACTEMENT CE QUI EST DANS LE PDF ⚠️⚠️⚠️

**NE MODIFIE RIEN. NE CORRIGE RIEN. NE RÉALIGNE RIEN.**

Tu dois faire une **COPIE PIXEL PAR PIXEL** de ce que tu vois dans le texte.

📋 **INSTRUCTIONS POUR LES TABLEAUX** :

1. **IDENTIFICATION DES EN-TÊTES (CRITIQUE)** :
   
   ⚠️⚠️⚠️ LES EN-TÊTES SONT LA PREMIÈRE LIGNE DU TABLEAU ⚠️⚠️⚠️
   
   **Comment reconnaître les EN-TÊTES** :
   - Ce sont des NOMS de colonnes (descriptifs, génériques)
   - Exemples d'en-têtes : "J", "Date", "Ménopur", "Anta", "E2", "LH", "Pg", "Right Ovary", "Left Ovary", "End"
   - Les en-têtes décrivent le TYPE d'information, PAS une valeur concrète
   - Généralement en gras ou dans une ligne séparée visuellement
   
   **Comment reconnaître les DONNÉES (PAS les en-têtes)** :
   - Ce sont des VALEURS concrètes
   - Exemples de données : "1", "2/10/25", "225UI", "350", "10/8"
   - Les données remplissent les colonnes définies par les en-têtes
   - Lignes qui viennent APRÈS la ligne d'en-têtes
   
   ❌ ERREUR CRITIQUE À ÉVITER :
   ```
   Ligne 1 : J, Date, Ménopur, Anta, E2, LH, Pg, Right Ovary, Left Ovary  ← EN-TÊTES
   Ligne 2 : J, Date, Ménopur, Anta, E2, LH, Pg, Right Ovary, Left Ovary  ← ❌ ERREUR ! Ce sont les MÊMES en-têtes répétés
   Ligne 3 : 1, 2/10/25, 225UI, /, 350, ...                                ← DONNÉES
   ```
   
   ✅ CORRECT :
   ```
   Ligne 1 : J, Date, Ménopur, Anta, E2, LH, Pg, Right Ovary, Left Ovary  ← EN-TÊTES (1 fois uniquement)
   Ligne 2 : 1, 2/10/25, 225UI, /, 350, ...                                ← DONNÉES ligne 1
   Ligne 3 : 2, 3/10/25, 225UI, /, 400, ...                                ← DONNÉES ligne 2
   ```
   
   **RÈGLE D'OR** :
   - Les en-têtes apparaissent UNE SEULE fois : au début du tableau
   - Si tu vois "J, Date, Ménopur..." deux fois de suite → la 2ème occurrence est une ERREUR
   - Utilise la PREMIÈRE occurrence comme noms de colonnes
   - Ignore la répétition

2. **EXTRACTION DES NOMS D'EN-TÊTES** :
   - Identifie la toute PREMIÈRE ligne du tableau
   - Extrais chaque nom de colonne EXACTEMENT comme il apparaît
   - Respecte les majuscules/minuscules
   - Respecte les espaces
   - N'ajoute rien, ne modifie rien
   
   Exemples de noms d'en-têtes corrects :
   - "J" (pas "j" ou "Jour")
   - "Date" (pas "date" ou "DATE")
   - "Right Ovary" (pas "RightOvary" ou "right_ovary")
   - "E2" (pas "e2" ou "Estradiol")

3. **COPIE LIGNE PAR LIGNE DES DONNÉES** :
   - Commence APRÈS la ligne d'en-têtes
   - Pour CHAQUE ligne de données :
     * Copie la valeur de CHAQUE cellule EXACTEMENT comme elle apparaît
     * Si une cellule contient "10/8" → copie "10/8" (ne sépare PAS)
     * Si une cellule contient "/" → copie "/"
     * Si une cellule est vide → mets null ou ""
     * Respecte l'ORDRE des colonnes défini par les en-têtes
   
3. **RÈGLES DE COPIE STRICTES** :
   
   ✅ CE QU'IL FAUT FAIRE :
   - Copier TEXTUELLEMENT chaque valeur
   - Garder les valeurs composées intactes : "10/8" reste "10/8"
   - Garder les unités avec les valeurs : "225UI" reste "225UI"
   - Copier "/" tel quel
   - Respecter la casse (majuscules/minuscules)
   
   ❌ CE QU'IL NE FAUT JAMAIS FAIRE :
   - Séparer "10/8" en deux valeurs distinctes
   - Modifier le format des dates
   - Corriger des fautes d'orthographe
   - Réorganiser les colonnes
   - Ajouter des colonnes qui n'existent pas
   - Supprimer des colonnes existantes
   - Interpréter ou déduire des valeurs manquantes

4. **STRUCTURE JSON POUR TABLEAU** :

Chaque ligne du tableau devient un objet JSON avec :
- Une clé pour CHAQUE colonne (nom de la colonne)
- La valeur EXACTE de la cellule

Exemple de tableau dans le PDF :
```
J    | Date     | E2  | Right Ovary | Left Ovary
1    | 2/10/25  | 350 | 10/8        | 
2    | 3/10/25  | 400 | 12          | 9
```

JSON attendu (copie EXACTE) :
```json
[
  {{
    "J": "1",
    "Date": "2/10/25",
    "E2": "350",
    "Right Ovary": "10/8",
    "Left Ovary": ""
  }},
  {{
    "J": "2",
    "Date": "3/10/25",
    "E2": "400",
    "Right Ovary": "12",
    "Left Ovary": "9"
  }}
]
```

⚠️ REMARQUE : "Right Ovary" contient "10/8" → on copie "10/8" tel quel, PAS deux valeurs séparées

5. **VALEURS MULTIPLES AVEC "/"** :

Si tu vois des valeurs comme "10/8", "225/150", etc. :

**OPTION 1 - Copie textuelle (RECOMMANDÉ)** :
```json
{{"right_ovary": "10/8"}}
```

**OPTION 2 - Si le "/" sépare clairement DEUX colonnes différentes** :
Exemple : si le tableau a visuellement deux sous-colonnes
```json
{{
  "right_ovary": "10",
  "left_ovary": "8"
}}
```

⚠️ MAIS SI TU N'ES PAS SÛR → utilise l'OPTION 1 (copie textuelle)

6. **INFORMATIONS HORS TABLEAU** :
   - Extrais aussi les informations qui ne sont PAS dans les tableaux
   - Copie-les textuellement
   - Organise-les par catégories logiques

**STRUCTURE JSON GLOBALE ATTENDUE** :

{{
  "informations_patient": {{
    "nom": "Valeur exacte du PDF",
    "age": "Valeur exacte",
    "date_naissance": "Valeur exacte"
  }},
  
  "biomarqueurs": {{
    "AMH": "Valeur exacte avec unité",
    "AFC": "Valeur exacte"
  }},
  
  "tableau_suivi": [
    {{
      "colonne1": "valeur exacte",
      "colonne2": "valeur exacte",
      "colonne3": "valeur exacte"
    }},
    {{
      "colonne1": "valeur exacte",
      "colonne2": "valeur exacte",
      "colonne3": "valeur exacte"
    }}
  ],
  
  "autres_informations": {{
    ...
  }}
}}

**VALIDATION FINALE** :
=======================

Avant de répondre, vérifie :
□ Chaque valeur du PDF est copiée EXACTEMENT ?
□ Aucune valeur n'a été modifiée ou "corrigée" ?
□ Les valeurs composées (ex: "10/8") sont intactes ?
□ L'ordre des colonnes est respecté ?
□ Toutes les lignes du tableau sont présentes ?
□ Les cellules vides sont marquées comme "" ou null ?

**EXEMPLES DE COPIE CORRECTE** :

PDF contient : "225UI"
✅ JSON : "225UI"
❌ JSON : {{"dose": 225, "unite": "UI"}} (trop d'interprétation)

PDF contient : "10/8"
✅ JSON : "10/8"
❌ JSON : [10, 8] (séparation non demandée)

PDF contient : "/"
✅ JSON : "/"
✅ JSON : null (si tu préfères pour les vides)

PDF contient : cellule vide
✅ JSON : ""
✅ JSON : null

Réponds UNIQUEMENT avec le JSON, sans ```json ni commentaires.
"""

        print("🤖 Envoi à Gemini pour copie exacte...")

        response = model.generate_content(prompt)
        result = response.text.strip()

        # Nettoyage du JSON
        if '```json' in result:
            result = result[result.find('```json')+7:result.rfind('```')].strip()
        elif '```' in result:
            result = result[result.find('```')+3:result.rfind('```')].strip()

        # Validation et parsing du JSON
        try:
            data = json.loads(result)
            print("✓ JSON valide !\n")
        except json.JSONDecodeError as e:
            print(f"⚠ Erreur JSON : {e}")
            print("\n🔄 Tentative de correction...\n")
            
            # Correction automatique
            correction_prompt = f"""
Le JSON suivant contient une erreur :

{result}

Erreur : {str(e)}

Corrige le JSON pour qu'il soit valide. Réponds UNIQUEMENT avec le JSON corrigé.
"""
            response = model.generate_content(correction_prompt)
            result = response.text.strip()
            
            if '```json' in result:
                result = result[result.find('```json')+7:result.rfind('```')].strip()
            elif '```' in result:
                result = result[result.find('```')+3:result.rfind('```')].strip()
            
            try:
                data = json.loads(result)
                print("✓ JSON corrigé !\n")
            except json.JSONDecodeError as e2:
                print(f"❌ Impossible de corriger : {e2}")
                with open(output_json.replace('.json', '_raw.txt'), 'w', encoding='utf-8') as f:
                    f.write(result)
                print(f"⚠ Sauvegardé en : {output_json.replace('.json', '_raw.txt')}")
                return None

        # Sauvegarde du JSON
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ JSON sauvegardé : {output_json}\n")

        # Affichage du résumé
        print("="*80)
        print("📊 DONNÉES EXTRAITES (COPIE EXACTE)")
        print("="*80 + "\n")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print("\n" + "="*80)

        return output_json

    except FileNotFoundError:
        print(f"❌ Fichier non trouvé : {pdf_path}")
        return None
    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# UTILISATION
# =============================================================================

""" if __name__ == "__main__":
    
    pdf_file = "sample.pdf"  # ⬅️ MODIFIEZ ICI
    
    print("🔵 EXTRACTION PDF → JSON (COPIE EXACTE)")
    print("="*80)
    
    # Extraction
    json_path = extract_pdf_to_json(pdf_file, "extraction_pdf.json")
    
    if json_path:
        print(f"\n✅ EXTRACTION TERMINÉE !")
        print(f"📁 Fichier JSON : {json_path}")
        print("\n💡 Le JSON contient une copie EXACTE du PDF")
        print("   • Aucune modification des valeurs")
        print("   • Aucune séparation de valeurs composées")
        print("   • Aucune correction ou réalignement")
    else:
        print("\n❌ ÉCHEC DE L'EXTRACTION")

     """

def extract_ivf_data_from_json(json_path):
    """
    Extrait les données IVF depuis la structure JSON spécifique du document médical
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extraction des informations patient
    info_patient = data.get('informations_patient', {})
    biomarqueurs = data.get('biomarqueurs', {})
    tableau_suivi = data.get('tableau_suivi', [])
    autres_info = data.get('autres_informations', {})
    
    # 1. Patient_id : Générer depuis le nom
    csv_exists = Path(r"C:\Users\yesmine\Desktop\Tanit\data\raw\patients.csv").exists()
    if csv_exists:
      with open("C:\\Users\\yesmine\\Desktop\\Tanit\\data\\raw\\patients.csv", 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        current_rows = list(all_lines)
        next_row_number = len(current_rows) + 1
    else:
       next_row_number = 1



    # Générer le Patient_id au format 25XXX
    patient_id = f"25{next_row_number:02d}" 
        



    # 2. Cycle_number : Extraire depuis "2nd", "1st", etc.
    cycle_str = info_patient.get('cycle_number', '1')
    cycle_number = 1
    if 'nd' in str(cycle_str).lower():
        cycle_number = 2
    elif 'rd' in str(cycle_str).lower():
        cycle_number = 3
    elif 'th' in str(cycle_str).lower():
        # Extraire le chiffre
        cycle_number = int(''.join(filter(str.isdigit, str(cycle_str))) or 1)
    elif 'st' in str(cycle_str).lower():
        cycle_number = 1
    
    # 3. Age : Calculer depuis la date de naissance
    birth_date_str = info_patient.get('birth_date', '')
    age = None
    if birth_date_str:
        try:
            # Format: "27/11/95"
            parts = birth_date_str.split('/')
            if len(parts) == 3:
                day, month, year = parts
                year = year.strip()
                # Ajouter le siècle
                if len(year) == 2:
                    year = '19' + year if int(year) > 50 else '20' + year
                birth_date = datetime.strptime(f"{day.strip()}/{month.strip()}/{year}", "%d/%m/%Y")
                age = (datetime.now() - birth_date).days // 365
        except:
            age = None
    
    # 4. Protocol : Mapper depuis "Flex Antago"
    protocol_str = info_patient.get('protocol', '').lower()
    if 'flex' in protocol_str and 'antag' in protocol_str:
        protocol = "flexible antagonist"
    elif 'fix' in protocol_str and 'antag' in protocol_str:
        protocol = "fixed antagonist"
    elif 'agon' in protocol_str:
        protocol = "agonist"
    else:
        protocol = None
    
    # 5. AMH : Depuis biomarqueurs
    amh = biomarqueurs.get('AMH')
    if amh:
        amh = f"{amh} ng/mL"
    
    # 6. N_Follicles : Depuis autres_informations
    n_follicles = autres_info.get('number_of_follicles')
    if n_follicles:
        n_follicles = int(n_follicles)
    
    # 7. E2_day5 : Chercher dans tableau_suivi le jour 5
    e2_day5 = None
    for row in tableau_suivi:
        j = str(row.get('J', '')).strip().replace('\n', '')
        if j == '5':
            e2_val = row.get('E2')
            if e2_val:
                e2_day5 = f"{e2_val} pg/mL"
            break
    
    # 8. AFC : Depuis biomarqueurs
    afc = biomarqueurs.get('AFC')
    
    # 9. Patient_Response : Depuis autres_informations
    response_str = autres_info.get('response', '').lower()
    if 'optimal' in response_str:
        patient_response = "optimal"
    elif 'low' in response_str or 'faible' in response_str or 'poor' in response_str:
        patient_response = "low"
    elif 'high' in response_str or 'élevé' in response_str or 'hyper' in response_str:
        patient_response = "high"
    else:
        patient_response = None
    
    # Construire l'objet final
    extracted_data = {
        'Patient_id': patient_id,
        'Cycle_number': cycle_number,
        'Age': age,
        'Protocol': protocol,
        'AMH': amh,
        'N_Follicles': n_follicles,
        'E2_day5': e2_day5,
        'AFC': afc,
        'Patient_Response': patient_response
    }
    
    return extracted_data


def add_json_to_csv(json_path, csv_path):
    """
    Lit les données IVF depuis un fichier JSON et les ajoute comme nouvelle ligne dans un CSV existant
    """
    print("\n" + "="*80)
    print("📥 EXTRACTION ET AJOUT DE DONNÉES JSON VERS CSV")
    print("="*80 + "\n")

    try:
        # 1. Lecture et extraction du fichier JSON
        print(f"📂 Lecture du JSON : {json_path}")
        
        extracted_data = extract_ivf_data_from_json(json_path)
        
        print(f"✓ Données extraites pour le patient : {extracted_data['Patient_id']}")
        
        # 2. Colonnes attendues
        fieldnames = [
            'Patient_id',
            'Cycle_number',
            'Age',
            'Protocol',
            'AMH',
            'N_Follicles',
            'E2_day5',
            'AFC',
            'Patient_Response'
        ]
        
        # 3. Vérifier si le CSV existe
        csv_exists = Path(csv_path).exists()
        
        if csv_exists:
            print(f"✓ CSV existant trouvé : {csv_path}")
            mode = 'a'  # Mode append
            write_header = False
        else:
            print(f"⚠ CSV non trouvé, création d'un nouveau : {csv_path}")
            mode = 'w'  # Mode write
            write_header = True
        
        # 4. Écriture dans le CSV
        print(f"\n✍️ Ajout de la ligne au CSV...")
        
        with open(csv_path, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Écrire l'en-tête seulement si nouveau fichier
            if write_header:
                writer.writeheader()
                print("✓ En-tête CSV créé")
            
            # Préparer la ligne avec valeurs vides si None
            row = {field: (extracted_data.get(field) if extracted_data.get(field) is not None else '') 
                   for field in fieldnames}
            
            writer.writerow(row)
            print(f"  ✓ Patient ajouté : {row.get('Patient_id', 'N/A')}")
        
        print(f"\n✅ DONNÉES AJOUTÉES AVEC SUCCÈS !")
        print(f"📁 Fichier CSV mis à jour : {csv_path}")
        
        # 5. Affichage récapitulatif
        print("\n📊 DONNÉES EXTRAITES ET AJOUTÉES :")
        print("-"*80)
        
        for key in fieldnames:
            value = extracted_data.get(key)
            display_value = value if value is not None else '❌ Non trouvé'
            print(f"  • {key:20s} : {display_value}")
        
        print("-"*80)
        
        # 6. Statistiques du CSV final
        print("\n📈 STATISTIQUES DU CSV :")
        print("-"*80)
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            total_patients = len(rows)
        
        print(f"  • Nombre total de patients dans le CSV : {total_patients}")
        
        print("-"*80 + "\n")
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ Fichier non trouvé : {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Erreur de lecture JSON : {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# UTILISATION
# =============================================================================

""" if __name__ == "__main__":
    
    # Chemins des fichiers
    json_file = "extraction_pdf.json"  # ⬅️ Fichier JSON avec structure médicale
    csv_file = "new_patients.csv"       # ⬅️ Fichier CSV où ajouter les données
    
    print("🔵 EXTRACTION ET AJOUT DE DONNÉES JSON VERS CSV")
    print("="*80)
    
    # Ajout des données
    success = add_json_to_csv(
        json_path=json_file,
        csv_path=csv_file
    )
    
    if success:
        print("\n✅ OPÉRATION TERMINÉE AVEC SUCCÈS !")
    else:
        print("\n❌ ÉCHEC DE L'OPÉRATION")

 """
    